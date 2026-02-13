# -*- coding: utf-8 -*-
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.lora.request import LoRARequest
from PIL import Image
import argparse
import asyncio
import uuid

import dotenv
dotenv.load_dotenv()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

async def run_conversation(args):
    # 1. Initialize the Async vLLM Engine
    print(f"--- Initializing vLLM Engine with model: {args.model} ---")
    print(f"--- LoRA enabled. Adapter path: {args.lora_path} ---")

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,  # Required for Qwen-VL models
        enable_lora=True,        # Must enable LoRA support at engine level
        max_lora_rank=64,        # Adjust based on your specific LoRA rank
        max_model_len=4096,      # Adjust based on GPU memory
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
        quantization="bitsandbytes"
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    processor = AutoProcessor.from_pretrained(args.model)

    lora_req = LoRARequest(
        "cq_adapter",
        1,
        args.lora_path
    )

    # sampling_params = SamplingParams(
    #     temperature=0,
    #     max_tokens=1024,
    #     top_k=-1,
    #     stop_token_ids=[]
    # )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        repetition_penalty=1.0,
        presence_penalty=2.0,
        max_tokens=1024,
        stop_token_ids=[],
        n=5,
        output_kind=RequestOutputKind.FINAL_ONLY
    )

    # Conversation history
    history = []
    history.append({
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an agent designed to ask clarifying questions to better understand a user's query."}
        ]
    })
    
    print("\n--- Engine Ready. Starting Conversation Loop ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            # 2. Get Inputs
            image_path_input = input("\n[User] Enter Image Path (or press Enter to skip): ").strip()
            
            # Allow exit
            if image_path_input.lower() in ['exit', 'quit']:
                break

            prompt_text = input("[User] Enter Prompt: ").strip()
            if prompt_text.lower() in ['exit', 'quit']:
                break

            if not prompt_text:
                print("Please enter a prompt.")
                continue

            # 3. Construct Message Payload
            # Qwen-VL expects specific content structure for images
            content_list = []

            # Handle Image Loading
            if image_path_input:
                if os.path.exists(image_path_input):
                    try:
                        # Load image for validation, but vLLM takes PIL or raw
                        # vLLM expects 'image' key in multi_modal_data
                        image_obj = Image.open(image_path_input).convert("RGB")
                        
                        # Add image to content list (Qwen-VL style)
                        content_list.append({"type": "image", "image": image_obj})
                        
                        # Pass data to vLLM engine
                        print(f" * Loaded image: {image_path_input}")
                    except Exception as e:
                        print(f" ! Error loading image: {e}")
                        continue
                else:
                    print(f" ! Path not found: {image_path_input}")
                    continue
            
            # Add text
            content_list.append({"type": "text", "text": prompt_text})

            # Append to history
            history.append({"role": "user", "content": content_list})

            request_id = str(uuid.uuid4())

            inputs = prepare_inputs_for_vllm(history, processor)
            results_generator = engine.generate(
                inputs,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_req
            )

            # Stream output
            print("\n[Assistant]: ", end="", flush=True)
            final_output = ""
            
            async for request_output in results_generator:
                # Get the newly generated text
                # generated_text = request_output.outputs[0].text
                
                # # Print only the new part (simple streaming logic)
                # # In robust apps, use logic to diff current vs previous output
                # print(generated_text[len(final_output):], end="", flush=True)
                # final_output = generated_text
                print(f"N-outputs: {len(request_output.outputs)}")

            # Now we print out all n outputs
            print("\n\n=========== ALL OUTPUTS ===========")
            for i, output in enumerate(request_output.outputs):
                print(f"Output {i}: {output.text}\n\n-----------\n\n")
            print("=====================================\n")

            # Append assistant response to history for context in next turn
            history.append({"role": "assistant", "content": final_output})

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Qwen-VL Async with LoRA")
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-32B-Instruct", 
                        help="HF Model ID (e.g. Qwen/Qwen3-VL-32B-Instruct)")
    
    parser.add_argument("--lora-path", type=str, default="./checkpoints/loras/qwen_3_vl_32b_clarification_lora_bfloat16/best_adapter", 
                        help="Local path to the LoRA adapter folder")

    args = parser.parse_args()

    asyncio.run(run_conversation(args))