from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from .cq_vllm_model import CQModelWorker
from .generic_vllm_model import GenericModelWorker
from .remote_cq_vllm_model import RemoteCQModel

class QwenModelInputProcessor:
    def __init__(self, model_cfg: dict):
        self.model_hf_transformers_key = model_cfg['model_hf_transformers_key']
        self.processor = AutoProcessor.from_pretrained(self.model_hf_transformers_key)

    def process_vision_info(self, messages):
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        return {
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs
        }

    def apply_chat_template(self, messages):
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_inputs(self, vision_info, templated_chat):
        return {
            'prompt': templated_chat,
            **vision_info
        }

    def prepare_inputs_for_vllm(self, messages):
        text = self.apply_chat_template(messages)
        vision_info = self.process_vision_info(messages)

        return self.format_inputs(vision_info, text)