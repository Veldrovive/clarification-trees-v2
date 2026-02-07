"""
My test script for dialog tree generation.
Keeps the frontiers of all trees and expands.

We can use asyncio with vLLM to keep the logic for each tree separate and add queries to the queue for each model
independently. This greatly simplifies the logic for managing the frontiers of all trees and batching input.
"""

from transformers import AutoProcessor
from clarification_trees.dialog_tree import DialogTree, NodeType, DialogTrajectory
from vllm import AsyncLLMEngine, SamplingParams
from qwen_vl_utils import process_vision_info
import hydra
from omegaconf import DictConfig
import uuid

class DialogTreeDFSManager:
    def __init__(self, dialog_tree: DialogTree):
        raise NotImplementedError("DialogTreeDFSManager is not implemented")

    def has_open_nodes(self) -> bool:
        raise NotImplementedError("DialogTreeDFSManager is not implemented")

    def get_next_node(self) -> tuple[DialogTrajectory, NodeType, int]:
        raise NotImplementedError("DialogTreeDFSManager is not implemented")

    def add_node(self, parent_node_id: int, output_text: str, output_node_type: NodeType) -> None:
        raise NotImplementedError("DialogTreeDFSManager is not implemented")


def prepare_inputs_for_qwen_vl_vllm(dialog_trajectory: DialogTrajectory, processor) -> dict:
    """
    Frome https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8
    """
    messages = dialog_trajectory.to_messages(model_name="qwen-3-vl")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

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


async def expand_tree(tree: DialogTree, cfg: DictConfig, cq_model: AsyncLLMEngine, answer_model: AsyncLLMEngine, processor, sampling_params: SamplingParams):
    dialog_tree_manager = DialogTreeDFSManager(tree)
    tree_uuid = str(uuid.uuid4())

    cq_node_types = set([NodeType.ROOT, NodeType.CLARIFYING_ANSWER])  # Node types that cause the tree to be expanded using the cq model
    answer_node_types = set([NodeType.CLARIFICATION_QUESTION])  # Node types that cause the tree to be expanded using the answer model


    while dialog_tree_manager.has_open_nodes():
        dialog_trajectory, input_node_type, node_id = dialog_tree_manager.get_next_node()
        node_uuid = tree_uuid + "_" + str(node_id)
        vllm_inputs = prepare_inputs_for_qwen_vl_vllm(dialog_trajectory, processor)

        if input_node_type in cq_node_types:
            # stream = cq_model.generate(vllm_inputs, sampling_params=sampling_params, request_id=node_uuid)
            engine = cq_model
            output_node_type = NodeType.CLARIFICATION_QUESTION
        elif input_node_type in answer_node_types:
            # stream = answer_model.generate(vllm_inputs, sampling_params=sampling_params, request_id=node_uuid)
            engine = answer_model
            output_node_type = NodeType.CLARIFYING_ANSWER
        else:
            raise ValueError(f"Unknown node type: {input_node_type}")

        stream = engine.generate(vllm_inputs, sampling_params=sampling_params, request_id=node_uuid)
        
        # TODO: Fix this to generate diverse and add all the nodes
        final_output = None
        async for request_output in stream:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(node_uuid)
                # Return or raise an error
                return False
        final_output = request_output
        dialog_tree_manager.add_node(node_id, final_output, output_node_type)

        
        
    

@hydra.main(config_path="src/clarification_trees/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    main()