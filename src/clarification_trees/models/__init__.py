from omegaconf import DictConfig
from pathlib import Path

from .transformers_model import TransformersModel

def construct_model(model_config: DictConfig, device: str, load_lora: bool = True, loras_path: Path | None = None) -> TransformersModel:
    """
    Construct a TransformersModel from the given config.
    """
    model = TransformersModel(model_config, device)
    if load_lora:
        assert loras_path is not None, "LORA path must be specified if load_lora is True"
        lora_config = model_config.lora_config
        if lora_config.use_lora:
            lora_id = lora_config.lora_id
            lora_path = loras_path / lora_id / "best_adapter"
            assert lora_path.exists(), f"LORA path {lora_path} does not exist"
            model.load_adapter(lora_path)
        else:
            print(f"WARNING: LORA is disabled for model {model_config.model_name}")
    return model