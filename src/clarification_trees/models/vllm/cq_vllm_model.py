import ray
import uuid
from pathlib import Path
import os

from transformers import BitsAndBytesConfig
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.lora.request import LoRARequest

@ray.remote(max_concurrency=64, num_cpus=1)
class CQModelWorker:
    model_name: str
    model_hf_transformers_key: str
    engine: AsyncLLMEngine
    max_new_tokens: int
    lora_config: dict
    use_lora: bool
    lora_id: str | None
    lora_rank: int | None
    adapter_path: Path | None
    lora_req: LoRARequest | None
    bnb_config: BitsAndBytesConfig | None
    
    def __init__(self, model_cfg: dict, loras_path: Path, quantize: bool = False, n_gpus: int = 1, batch_per_gpu: int = 8):
        self.model_cfg = model_cfg
        self.model_name = model_cfg['model_name']
        self.model_hf_transformers_key = model_cfg['model_hf_transformers_key']
        self.max_new_tokens = model_cfg['max_new_tokens']

        print(f"CQ worker started with access to GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

        self.lora_config = model_cfg['lora_config']
        self.use_lora = self.lora_config['use_lora']
        if self.use_lora:
            self.lora_id = self.lora_config['lora_id']
            self.lora_rank = self.lora_config['peft_config']['r']
            self.adapter_path = loras_path / self.lora_id / "best_adapter"
            assert self.adapter_path.exists(), f"LoRA adapter not found at {self.adapter_path}"
            self.lora_req = LoRARequest(
                "cq_adapter",
                1,
                self.adapter_path.absolute().as_posix()
            )
        else:
            self.lora_id = None
            self.lora_rank = None
            self.adapter_path = None
            self.lora_req = None

        if "sampling_params" in model_cfg:
            self.sampling_params = SamplingParams(**model_cfg["sampling_params"], output_kind=RequestOutputKind.FINAL_ONLY)
        else:
            self.sampling_params = SamplingParams(
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

        if "bnb_config" in model_cfg:
            self.bnb_config = BitsAndBytesConfig(**model_cfg["bnb_config"])
        else:
            self.bnb_config = None

        engine_args = AsyncEngineArgs(
            model=self.model_hf_transformers_key,
            trust_remote_code=True,  # Required for Qwen-VL models
            enable_lora=self.use_lora,        # Must enable LoRA support at engine level
            max_lora_rank=self.lora_rank or 64,        # Adjust based on your specific LoRA rank
            max_model_len=4096,      # Adjust based on GPU memory
            gpu_memory_utilization=0.85,
            disable_log_stats=False,
            tensor_parallel_size=n_gpus,
            # data_parallel_size=n_gpus,
            max_num_seqs=batch_per_gpu * n_gpus,
            quantization="bitsandbytes" if quantize else None
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, inputs, request_id: str | None = None, n_outputs: int = 1):
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Set the sampling params
        self.sampling_params.n = n_outputs

        results_generator = self.engine.generate(
            inputs,
            sampling_params=self.sampling_params,
            request_id=request_id,
            lora_request=self.lora_req
        )

        async for request_output in results_generator:
            # This just waits for the request to complete
            pass
        
        return request_output

    async def ping(self):
        return True