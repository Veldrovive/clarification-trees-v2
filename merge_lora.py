import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import dotenv
dotenv.load_dotenv()

from clarification_trees.models import construct_model

@hydra.main(config_path="src/clarification_trees/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    lora_checkpoint_path = Path(cfg.paths.checkpoints.loras)

    model_config = cfg.clarification_model
    print(f"Merging lora for model {model_config}")
    
    model = construct_model(
        model_config,
        device=cfg.devices.clarification,
        load_lora=True,
        loras_path=lora_checkpoint_path,
        allow_quantization=False,
    )
    assert model.adapted_model is not None, "No adapter is currently loaded or constructed."
    model.adapted_model.eval()

    lora_config = model_config.get("lora_config", None)
    adapter_id = lora_config.lora_id
    adapter_path = lora_checkpoint_path / adapter_id / "best_adapter"
    
    merged_model_path = adapter_path / "merged_model"
    merged_model_path.mkdir(parents=True, exist_ok=True)
    
    model.merge_and_save_adapter(merged_model_path)

if __name__ == "__main__":
    main()
