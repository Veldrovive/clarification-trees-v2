from omegaconf import DictConfig
from pathlib import Path

from .transformers_model import TransformersModel
from .semantic_clustering import SemanticClusterer, BidirectionalEntailmentClusterer, HybridClusterer, Clusterer

def construct_model(model_config: DictConfig, device: str, load_lora: bool = True, loras_path: Path | None = None) -> TransformersModel:
    """
    Construct a TransformersModel from the given config.
    """
    model = TransformersModel(model_config, device)
    if load_lora:
        assert loras_path is not None, "LORA/Adapter path must be specified if load_lora is True"
        
        lora_config = model_config.get("lora_config", None)
        pt_config = model_config.get("prompt_tuning_config", None)

        if lora_config and lora_config.get("use_lora", False):
            adapter_id = lora_config.lora_id
            adapter_path = loras_path / adapter_id / "best_adapter"
            assert adapter_path.exists(), f"LoRA path {adapter_path} does not exist"
            model.load_adapter(adapter_path)
            
        elif pt_config and pt_config.get("use_prompt_tuning", False):
            adapter_id = pt_config.prompt_tuning_id
            adapter_path = loras_path / adapter_id / "best_adapter"
            assert adapter_path.exists(), f"Prompt Tuning path {adapter_path} does not exist"
            model.load_adapter(adapter_path)
            
        else:
            print(f"WARNING: No Adapter (LoRA or Prompt Tuning) enabled for model {model_config.model_name}")
    return model

def construct_semantic_clusterer(semantic_cluster_config: DictConfig, device: str) -> Clusterer:
    """
    Construct a Clusterer from the given config.
    """
    if semantic_cluster_config.model_type == "sentence_transformers":
        return SemanticClusterer(semantic_cluster_config, device)
    elif semantic_cluster_config.model_type == "bidirectional_entailment":
        return BidirectionalEntailmentClusterer(semantic_cluster_config, device)
    elif semantic_cluster_config.model_type == "hybrid":
        return HybridClusterer(semantic_cluster_config, device)
    else:
        raise NotImplementedError(f"Model type {semantic_cluster_config.model_type} is not implemented for semantic clustering")