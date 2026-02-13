import os
import ray
import asyncio
from pathlib import Path
from vllm import RequestOutput
import hydra
import json
from typing import Any
import glob
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from PIL import ImageOps, Image

from clarification_trees.dataset import ClearVQADataset
from clarification_trees.models.vllm import QwenModelInputProcessor, CQModelWorker

os.environ["NCCL_P2P_DISABLE"] = "1"

# Initialize Ray (if running locally, this auto-detects GPUs)
ray.init(ignore_reinit_error=True)

async def process_partition(
    partition_id: int,
    indices: list[int],
    dataset: Any,
    worker_handle: ray.ObjectRef,
    processor: QwenModelInputProcessor,
    system_prompt: str,
    output_dir: Path,
    pbar: tqdm
):
    """
    This is the sub-function that takes a range (indices) and processes them
    one at a time, waiting for each to finish before moving to the next.
    """
    
    # Create a temporary file for this specific partition to avoid write conflicts
    partition_file = output_dir / f"results_part_{partition_id}.jsonl"
    
    # We open the file in append mode
    with open(partition_file, "w", encoding="utf-8") as f:
        
        for idx in indices:
            try:
                # 1. Prepare Data
                img, data = dataset[idx]
                ambiguous_question = data["blurred_question"]

                # Resize and pad images to a smaller square
                new_size = 512
                img = ImageOps.fit(img, (new_size, new_size), Image.Resampling.LANCZOS)
                
                
                # Construct messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": ambiguous_question},
                            {"type": "image", "image": img}
                        ]
                    }
                ]
                
                # 2. Preprocess (CPU side)
                # Note: If this is very CPU heavy, it might block the event loop slightly.
                # For heavy preprocessing, one might wrap this in asyncio.to_thread
                inputs = processor.prepare_inputs_for_vllm(messages)

                # 3. Generate (Remote GPU side)
                # Call remote worker
                request_ref = worker_handle.generate.remote(inputs, n_outputs=5)
                
                # 4. Await Result (The "Wait for finish" step)
                request_output: RequestOutput = await request_ref
                # request_output: RequestOutput = await asyncio.to_thread(ray.get, request_ref)
                
                # 5. Process Output
                generated_texts = [o.text for o in request_output.outputs]
                
                result_record = {
                    "dataset_index": idx,
                    "ambiguous_question": ambiguous_question,
                    "gold_answer": data.get("gold_answer"),
                    "generated_clarifications": generated_texts,
                    "prompt_token_ids": request_output.prompt_token_ids,
                    "finished": request_output.finished,
                }
                
                # 6. Save immediately
                f.write(json.dumps(result_record) + "\n")
                f.flush() # Ensure it's written to disk
                
            except Exception as e:
                print(f"\nError processing index {idx} in partition {partition_id}: {e}")
                # Log error but continue loop
                error_record = {"dataset_index": idx, "error": str(e)}
                f.write(json.dumps(error_record) + "\n")
            
            finally:
                # Update the shared progress bar
                pbar.update(1)

async def run_dataset_processing(cfg: DictConfig):
    """
    Orchestrator function.
    """
    # 1. Setup Configuration
    model_config = cfg.clarification_model
    model_config_dict = OmegaConf.to_container(model_config, resolve=True)
    lora_checkpoint_path = Path(cfg.paths.checkpoints.loras)
    
    output_dir = Path("output_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    final_output_file = output_dir / "final_results.jsonl"

    # 2. Initialize Resources
    print("Initializing Dataset...")
    ds = ClearVQADataset()
    total_samples = len(ds)
    # Optional: limit samples for testing
    # total_samples = 20 
    
    print("Initializing Ray Worker...")
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # Necessary to prevent Ray from overriding CUDA_VISIBLE_DEVICES to empty
    cq_worker = CQModelWorker.options(
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "6,7"}},
        num_gpus=2
    ).remote(model_config_dict, lora_checkpoint_path, n_gpus=2, batch_per_gpu=12, quantize=False)
    
    # Initialize Processor locally (it's lightweight enough usually, or move to actor if heavy)
    cq_model_input_processor = QwenModelInputProcessor(model_config_dict)
    
    base_system_prompt = cfg.clarification_model.base_prompt

    # 3. Define Concurrency (Loops running simultaneously)
    # This determines how many "streams" of requests happen at once.
    num_concurrent_loops = 100
    
    # Calculate chunks
    chunk_size = (total_samples + num_concurrent_loops - 1) // num_concurrent_loops
    
    tasks = []
    
    # Create a shared progress bar
    pbar = tqdm(total=total_samples, desc="Processing Dataset", unit="sample")
    
    print(f"Starting {num_concurrent_loops} concurrent loops processing {total_samples} items...")

    # 4. Create Tasks
    for i in range(num_concurrent_loops):
        start_idx = i * chunk_size
        if start_idx >= total_samples:
            break
        end_idx = min(start_idx + chunk_size, total_samples)
        
        # Create the list of indices for this specific loop
        indices = list(range(start_idx, end_idx))
        
        # Schedule the coroutine
        tasks.append(
            process_partition(
                partition_id=i,
                indices=indices,
                dataset=ds,
                worker_handle=cq_worker,
                processor=cq_model_input_processor,
                system_prompt=base_system_prompt,
                output_dir=output_dir,
                pbar=pbar
            )
        )

    # 5. Run all loops concurrently
    await asyncio.gather(*tasks)
    pbar.close()

    # 6. Merge Results
    print("Processing complete. Merging files...")
    with open(final_output_file, 'w', encoding='utf-8') as outfile:
        for fname in glob.glob(str(output_dir / "results_part_*.jsonl")):
            with open(fname, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
            # Optional: Delete part file after merge
            # os.remove(fname)
            
    print(f"Saved merged results to {final_output_file}")

async def async_main(cfg: DictConfig):
    model_config = cfg.clarification_model
    model_config_dict = OmegaConf.to_container(model_config, resolve=True)
    lora_checkpoint_path = Path(cfg.paths.checkpoints.loras)

    ds = ClearVQADataset()
    img, data = ds[0]
    ambiguous_question = data["blurred_question"]
    img_caption = data["caption"]
    unambiguous_question = data["question"]
    gold_answer = data["gold_answer"]
    answers = data["answers"]
    clarifying_question = data["clarification_question"]

    print(f"Ambiguous question: {ambiguous_question}")
    print(f"Image caption: {img_caption}")
    print(f"Unambiguous question: {unambiguous_question}")
    print(f"Gold answer: {gold_answer}")
    print(f"Answers: {answers}")
    print(f"Clarifying question: {clarifying_question}")

    cq_worker = CQModelWorker.remote(model_config_dict, lora_checkpoint_path)
    cq_model_input_processor = QwenModelInputProcessor(model_config_dict)

    base_system_prompt = cfg.clarification_model.base_prompt
    
    test_messages = [
        {
            "role": "system",
            "content": base_system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": ambiguous_question
                },
                {
                    "type": "image",
                    "image": img
                }
            ]
        }
    ]
    inputs = cq_model_input_processor.prepare_inputs_for_vllm(test_messages)
    request_output = cq_worker.generate.remote(inputs, n_outputs=20)
    
    # Wait for the request to complete
    request_output: RequestOutput = await request_output
    
    # Print the output
    print(f"======== {len(request_output.outputs)} outputs ========")
    for i, output in enumerate(request_output.outputs):
        print(f"Output {i}: {output.text}")
        print(f"\n--------\n")

    

@hydra.main(config_path="src/clarification_trees/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # asyncio.run(async_main(cfg))
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        
    try:
        asyncio.run(run_dataset_processing(cfg))
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        ray.shutdown()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        ray.shutdown()
        raise

if __name__ == "__main__":
    main()