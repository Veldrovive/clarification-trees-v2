"""
Manages a subprocess that runs a vLLM server
"""

from openai.types.chat import ChatCompletion
from omegaconf import DictConfig
from pathlib import Path
import requests
import os
import subprocess
import asyncio
import time

from openai import AsyncOpenAI

class RemoteCQModel:
    """
    A wrapper around a remote vLLM server that can be used to generate clarifications.
    """

    process: subprocess.Popen | None
    client: AsyncOpenAI
    is_running: bool

    def __init__(
        self,
        model_cfg: DictConfig, loras_path: Path,
        quantize: bool = False,
        gpus: list[int] = [0],
        batch_per_gpu: int = 8,
        port: int = 29001,
        environment_path: Path | None = None,
        startup_timeout: int = 120,
        log_file: Path | None = None,
        debug: bool = False
    ):
        self.model_cfg = model_cfg

        self.model_hf_transformers_key = model_cfg['model_hf_transformers_key']
        self.lora_config = model_cfg['lora_config']
        if "sampling_params" in model_cfg:
            self.sampling_params = model_cfg['sampling_params']
        else:
            self.sampling_params = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 40,
                "repetition_penalty": 1.0,
                "presence_penalty": 2.0,
                "max_tokens": 1024,
                "stop_token_ids": []
            }
        self.loras_path = loras_path
        self.quantize = quantize
        self.gpus = gpus
        self.batch_per_gpu = batch_per_gpu
        self.port = port
        self.environment_path = environment_path  # Like "./venv"
        self.startup_timeout = startup_timeout
        self.log_file = log_file
        self.debug = debug

        self.process = None
        # If there is already a server running on this port, then we just use that one.
        self.is_running = self.check_health()
        if self.is_running:
            print(f"vLLM server already running on port {self.port}. Skipping startup.")
        else:
            print(f"No vLLM server running on port {self.port}. Server needs to be started manually. Call start_server() to start the server.")

        self.client = self._get_openai_client()

    def _get_base_url(self):
        return f"http://localhost:{self.port}"

    def _get_openai_client(self):
        return AsyncOpenAI(base_url=self._get_base_url() + "/v1", api_key="EMPTY")

    def check_health(self):
        try:
            response = requests.get(f"{self._get_base_url()}/health")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    async def start_server(self):
        if self.is_running:
            return

        python_executable = "python"
        if self.environment_path is not None:
            python_executable = str(self.environment_path / "bin" / "python")


        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        if self.debug:
            env["VLLM_LOGGING_LEVEL"] = "DEBUG"
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            env["NCCL_DEBUG"] = "TRACE"

        self.model_key = self.model_hf_transformers_key
        command = [
            # python_executable,
            # "-m",
            "vllm", "serve", self.model_key,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            # "--max-cudagraph-capture-size", "8192",
            "--trust-remote-code",
            "--tensor-parallel-size", str(len(self.gpus)),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85",
            "--allowed-local-media-path", "/",
            # "--max-num-seqs", str(self.batch_per_gpu * len(self.gpus)),
        ]

        if self.quantize:
            command.extend(["--quantization", "bitsandbytes"])

        self.use_lora = self.lora_config['use_lora']
        if self.use_lora:
            print("Using LoRA")
            self.lora_id = self.lora_config['lora_id']
            self.lora_rank = self.lora_config['peft_config']['r']
            self.adapter_path = self.loras_path / self.lora_id / "best_adapter"
            merged_adapter_path = self.adapter_path / "merged_model"
            if merged_adapter_path.exists():
                self.model_key = merged_adapter_path.absolute().as_posix()
                command[2] = self.model_key
                print(f"Using merged adapter: {merged_adapter_path}")
            else:
                assert self.adapter_path.exists(), f"LoRA adapter not found at {self.adapter_path}"
                self.model_key = "cq_adapter"
                command.extend([
                    "--enable-lora",
                    "--max-lora-rank", str(self.lora_rank),
                    "--lora-modules", f"cq_adapter={self.adapter_path.absolute().as_posix()}",
                    "--lora-dtype", "bfloat16"
                ])
        else:
            print("Not using LoRA")
        
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as log_file:
                self.process = subprocess.Popen(command, env=env, stdout=log_file, stderr=log_file)
        else:
            self.process = subprocess.Popen(command, env=env)

        print(f"Running command: {' '.join(command)}")

        print(f"Waiting for vLLM server to start on port {self.port}...")
        start_time = time.time()
        while not self.check_health():
            if time.time() - start_time > self.startup_timeout:
                raise TimeoutError(f"vLLM server did not start within {self.startup_timeout} seconds")
            await asyncio.sleep(1)
        
        self.is_running = True
        print("vLLM server started successfully")

    def stop_server(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.is_running = False
            print("vLLM server stopped")
        else:
            print("No vLLM server running to stop")

    async def generate(self, messages, n_outputs: int = 1) -> ChatCompletion:
        sampling_params = {
            **self.sampling_params,
            "n": n_outputs
        }

        # TODO: Figure out how to check if we have a LoRA adapter loaded if the server is external.
        # model = "cq_adapter" if self.use_lora else self.model_hf_transformers_key

        response = await self.client.chat.completions.create(
            model=self.model_key,
            messages=messages,
            extra_body=sampling_params
        )

        return response


async def run_test():
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from dotenv import load_dotenv
    load_dotenv()

    with initialize(version_base=None, config_path="../../config", job_name="test_app"):
        config = compose(config_name="config")
    
    lora_checkpoint_path = Path(config.paths.checkpoints.loras)
    
    remote_cq_model = RemoteCQModel(
        config.clarification_model,
        lora_checkpoint_path,
        environment_path=Path("/scratch4/home/adempst/projects/clarification-trees-v2/venv_vllm"),
        gpus=[7],
        log_file=Path("cq_vllm_server.log")
    )
    await remote_cq_model.start_server()
    res = await remote_cq_model.generate([{"role": "user", "content": "Hello"}], 5)
    for i, choice in enumerate(res.choices):
        print(f"Choice {i}: {choice.message.content}")
    remote_cq_model.stop_server()

if __name__ == "__main__":
    asyncio.run(run_test())
    