from omegaconf import DictConfig
import streamlit as st
from pathlib import Path

from clarification_trees.models import construct_model
from clarification_trees.dataset import ClearVQADataset

@st.cache_resource(show_spinner=True)
def load_clarification_model(_cfg: DictConfig):
    """
    Loads the clarifying question model and the corresponding LORA.
    """
    lora_checkpoint_path = Path(_cfg.paths.checkpoints.loras)

    model_config = _cfg.clarification_model
    
    model = construct_model(
        model_config,
        device=_cfg.devices.clarification,
        load_lora=True,
        loras_path=lora_checkpoint_path,
    )
    assert model.adapted_model is not None, "No adapter is currently loaded or constructed."
    model.adapted_model.eval()

    return model

@st.cache_resource(show_spinner=True)
def load_answer_model(_cfg: DictConfig):
    """
    Loads the answer model.
    """
    lora_checkpoint_path = Path(_cfg.paths.checkpoints.loras)
    
    model_config = _cfg.answer_model
    
    model = construct_model(
        model_config,
        device=_cfg.devices.answer,
        load_lora=True,
        loras_path=lora_checkpoint_path,
    )
    model.base_model.eval()

    return model

@st.cache_resource
def load_clearvqa_dataset(_cfg: DictConfig):
    dataset = ClearVQADataset(table_name="val_annotated.jsonl")
    return dataset