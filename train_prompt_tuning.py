import dotenv
dotenv.load_dotenv()

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
import hydra
from pathlib import Path
from tqdm import tqdm
import wandb
import gc

from omegaconf import DictConfig, OmegaConf

from clarification_trees.models import TransformersModel, construct_model
from clarification_trees.dialog_tree import DialogTree, NodeType
from clarification_trees.utils import set_seed
from clarification_trees.dataset import ClearVQADataset

from logging import getLogger
logger = getLogger(Path(__file__).name)

def get_collate_fn(model: TransformersModel):
    def clarification_sample_collate(batch):
        processed_samples = []
        
        for sample in batch:
            image = sample[0]
            ambiguous_question = sample[1]["blurred_question"]
            clarifying_question = sample[1]["clarification_question"]
            
            tree = DialogTree(ambiguous_question, image)
            cq = tree.add_node(DialogTree.ROOT, NodeType.CLARIFICATION_QUESTION, None, clarifying_question)
            trajectory = tree.get_trajectory(cq)

            tokenized = model.preprocess_training_inputs(trajectory)
            processed_samples.append(tokenized)
    
        input_ids = [s["input_ids"] for s in processed_samples]
        labels = [s["labels"] for s in processed_samples]

        pad_token_id = model.processor.tokenizer.pad_token_id

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        attention_mask = [s["attention_mask"] for s in processed_samples]
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        pixel_values = torch.stack([s["pixel_values"] for s in processed_samples])
        grid_thw = torch.stack([s["image_grid_thw"] for s in processed_samples])

        batch = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw
        }

        return batch

    return clarification_sample_collate

def evaluate(model: TransformersModel, val_loader: DataLoader, device: str, step_id: int):
    assert model.adapted_model is not None
    model.adapted_model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation Loss")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)

            outputs = model.adapted_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            progress.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss:.4f}")

    wandb.log({"val/loss": avg_loss, "val/step": step_id})

    model.adapted_model.train()
    return avg_loss

def generate_samples(model: TransformersModel, val_loader: DataLoader, device: str, step_id: int, n_samples: int = 20):
    assert model.adapted_model is not None
    model.adapted_model.eval()

    logger.info(f"Generating {n_samples} samples for step {step_id}")

    table_data = []
    columns = ["Image Index", "Image", "Question Id", "Ambiguous Question", "Ground Truth CQ", "Model Prediction", "Answer"]

    with torch.no_grad():
        vqa_dataset = val_loader.dataset
        progress = tqdm(range(min(n_samples, len(vqa_dataset))), desc="Generating Samples")
        for i in progress:
            sample = vqa_dataset[i]
            image = sample[0]
            ambiguous_q = sample[1]["blurred_question"]
            gt_clarification = sample[1]["clarification_question"]
            question_id = sample[1]["question_id"]
            answer = sample[1]["gold_answer"]

            tree = DialogTree(ambiguous_q, image)
            trajectory = tree.get_trajectory(DialogTree.ROOT)

            # Note: For soft prompts, PEFT handles the virtual tokens automatically 
            # when calling generate if the adapter is active.
            prediction = model.generate(trajectory)
            prediction_text = prediction[0] if isinstance(prediction, list) else prediction
            
            table_data.append([
                i,
                wandb.Image(image),
                question_id,
                ambiguous_q,
                gt_clarification,
                prediction_text,
                answer
            ])

    wandb.log({"val/predictions": wandb.Table(data=table_data, columns=columns), "val/step": step_id})

    model.adapted_model.train()

def save_checkpoint(
    save_dir: Path,
    model: TransformersModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step_id: int,
    epoch: int,
    is_best: bool = False
):
    checkpoint_dir = save_dir / f"epoch_{epoch:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save the adapter (soft prompt weights)
    model.save_adapter(checkpoint_dir / "adapter")

    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": step_id,
        "epoch": epoch,
        "is_best": is_best
    }, checkpoint_dir / "state.pt")

    if is_best:
        model.save_adapter(save_dir / "best_adapter")

def train_loop(model: TransformersModel, train_loader: DataLoader, val_loader: DataLoader, cfg: DictConfig):
    model_config = cfg.clarification_model
    # Switch to reading prompt_tuning_config
    pt_config = model_config.prompt_tuning_config

    training_config = pt_config.training_config
    # peft_config is accessed during model construction, not needed here directly

    # Training config
    set_seed(training_config.seed)
    epochs = training_config.epochs
    evaluate_first = training_config.evaluate_first
    device = training_config.device
    lr = training_config.lr
    weight_decay = training_config.weight_decay
    gradient_accumulation_steps = training_config.gradient_accumulation_steps
    max_grad_norm = training_config.max_grad_norm
    warmup_ratio = training_config.warmup_ratio
    patience = training_config.patience

    # Get the save dir
    checkpoint_path = Path(cfg.paths.checkpoints.loras) # We can reuse the path or create a separate one
    pt_id = pt_config.prompt_tuning_id
    save_dir = checkpoint_path / pt_id
    
    if save_dir.exists():
        logger.warning(f"Checkpoint directory {save_dir} already exists. Overwrite?")
        if not input("Overwrite? (y/n): ").lower() == "y":
            return
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving Prompt Tuning checkpoints to {save_dir}")
    
    # Optimizer: Only the soft prompt tokens are trainable
    assert model.adapted_model is not None, "No adapter is currently loaded or constructed."
    trainable_params = [p for p in model.adapted_model.parameters() if p.requires_grad]
    
    # Sanity check
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Number of trainable parameters (Soft Prompt): {num_trainable}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    logger.info(f"Starting Prompt Tuning: {epochs} epochs, {len(train_loader)} batches/epoch")
    global_step = 0
    
    best_val_loss = float("inf")
    best_val_loss_epoch = -1
    
    if evaluate_first:
        logger.info("Evaluating before training...")
        best_val_loss = evaluate(model, val_loader, device, global_step)
        generate_samples(model, val_loader, device, global_step)

    for epoch in range(epochs):
        model.adapted_model.train()
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        # Ensure gradients are enabled for the soft prompt
        # PEFT handles this, but sometimes with gradient checkpointing on base models 
        # extra care is needed. Usually prepare_model_for_kbit_training handles `enable_input_require_grads`.

        epoch_loss = 0.0

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)

            outputs = model.adapted_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/lr": current_lr,
                    "train/step": global_step,
                    "train/epoch": epoch + (step / len(train_loader))
                })
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
        logger.info(f"End of Epoch {epoch+1}. Running validation...")
        val_loss = evaluate(model, val_loader, device, global_step)
        generate_samples(model, val_loader, device, global_step)

        if val_loss < best_val_loss:
            logger.info(f"New best validation loss: {val_loss}")
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            save_checkpoint(save_dir, model, optimizer, scheduler, global_step, epoch, is_best=True)
        else:
            logger.info(f"Validation loss did not improve from {best_val_loss}")
            save_checkpoint(save_dir, model, optimizer, scheduler, global_step, epoch, is_best=False)
            if epoch - best_val_loss_epoch >= patience:
                logger.info(f"No improvement for {patience} epochs. Early stopping.")
                break


def construct_model_with_soft_prompt(model_config) -> TransformersModel:
    pt_config = model_config.prompt_tuning_config
    peft_config = pt_config.peft_config
    training_config = pt_config.training_config
    device = training_config.device

    logger.info("Loading base model for Prompt Tuning...")
    # Load base model, disabling existing adapters loading
    model = construct_model(model_config, device, load_lora=False)
    
    assert isinstance(model, TransformersModel)
    
    print("Adding Soft Prompt (Prompt Tuning) to model...")
    # Construct the adapter using the new generic method
    model.construct_new_adapter(peft_config, adapter_type="prompt_tuning")
    
    assert model.adapted_model is not None, "No adapter was constructed."
    
    train_p, tot_p = model.adapted_model.get_nb_trainable_parameters()
    print(f'Trainable parameters:      {train_p}')
    print(f'Total parameters:          {tot_p/1e6:.2f}M')
    print(f'% of trainable parameters: {100*train_p/tot_p:.4f}%')

    return model

@hydra.main(config_path="src/clarification_trees/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_config = cfg.clarification_model
    pt_config = model_config.prompt_tuning_config
    
    assert pt_config.use_prompt_tuning, "Prompt tuning must be enabled in config for this script"
    
    training_config = pt_config.training_config
    pt_id = pt_config.prompt_tuning_id

    logger.info("Starting training for Clarification Soft Prompt")
    logger.info(f"Model config: {model_config}")

    model = construct_model_with_soft_prompt(model_config)
    collate_fn = get_collate_fn(model)

    train_ds = ClearVQADataset(table_name="train_annotated.jsonl")
    val_ds = ClearVQADataset(table_name="val_annotated.jsonl")

    # Artificially limit the dataset for debugging if needed
    from torch.utils.data import Subset
    # train_ds = Subset(train_ds, range(100))
    val_ds = Subset(val_ds, range(1000))

    train_loader = DataLoader(
        train_ds, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    wandb.init(
        project="clarification-cold-start",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=pt_id,
        tags=["prompt-tuning"]
    )

    train_loop(model, train_loader, val_loader, cfg)

    wandb.finish()

if __name__ == "__main__":
    main()