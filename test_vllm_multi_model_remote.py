import asyncio
from pathlib import Path
import hydra
import json
from typing import Any
import glob
from tqdm import tqdm
from omegaconf import DictConfig
from PIL import ImageOps, Image
from openai.types.chat import ChatCompletion
import tempfile

import dotenv
dotenv.load_dotenv()

from clarification_trees.dataset import ClearVQADataset
from clarification_trees.models.vllm import RemoteCQModel

async def process_partition(
    partition_id: int,
    indices: list[int],
    dataset: Any,
    remote_cq_model: RemoteCQModel,
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
                # Save to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    img.save(temp_file.name)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": ambiguous_question},
                                # {"type": "image", "image": img, "uuid": data["question_id"]}
                                {"type": "image_url", "image_url": {"url": f"file://{temp_file.name}"}, "uuid": data["question_id"]}
                            ]
                        }
                    ]
                
                # 2. Preprocess (CPU side)
                # Note: If this is very CPU heavy, it might block the event loop slightly.
                # For heavy preprocessing, one might wrap this in asyncio.to_thread
                # inputs = processor.prepare_inputs_for_vllm(messages)

                # 3. Generate (Remote GPU side)
                # Call remote worker
                request_ref = remote_cq_model.generate(messages, n_outputs=5)
                
                # 4. Await Result (The "Wait for finish" step)
                request_output: ChatCompletion = await request_ref
                
                # 5. Process Output
                generated_texts = [o.message.content for o in request_output.choices]
                
                result_record = {
                    "dataset_index": idx,
                    "ambiguous_question": ambiguous_question,
                    "gold_answer": data.get("gold_answer"),
                    "generated_clarifications": generated_texts,
                    "finish_reason": request_output.choices[0].finish_reason,
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

    remote_cq_model = RemoteCQModel(model_config, lora_checkpoint_path, log_file=Path("cq_model.log"), startup_timeout=180, gpus=[4,5,6,7], debug=False)
    await remote_cq_model.start_server()
    
    base_system_prompt = cfg.clarification_model.answer_base_prompt

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
                remote_cq_model=remote_cq_model,
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
    

@hydra.main(config_path="src/clarification_trees/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        asyncio.run(run_dataset_processing(cfg))
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise

if __name__ == "__main__":
    main()