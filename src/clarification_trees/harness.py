"""
Class. Holds the models and configuration required to roll out dialog trees.
Provides helper methods to run.
"""

from typing import List, Optional
from omegaconf import DictConfig
import wandb
from PIL import Image
from enum import Enum
from clarification_trees.models import TransformersModel
from clarification_trees.models import construct_model


from clarification_trees.utils import check_git_commit, get_git_commit, set_seed
from clarification_trees.dialog_tree import DialogTree, NodeType

class DialogTreeHarness:
    def __init__(self,
        cfg: DictConfig, logger,
        clarification_model: TransformersModel,
        clarifying_model: TransformersModel,
        inference_model: TransformersModel
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = cfg.training.device
        self.batch_size = cfg.training.batch_size

        self.clarification_model = clarification_model
        self.clarifying_model = clarifying_model
        self.inference_model = inference_model
        
        set_seed(cfg.training.seed)

    @staticmethod
    def from_config(cfg: DictConfig, logger):
        # Start by constructing the models
        pass
        
    # def rollout_dialog_tree(
    #     self,
    #     images: List[Image.Image],
    #     blurred_questions: List[str], clear_questions: List[str],
    #     gold_clarification_questions: List[str],
    #     captions: List[str], ambiguity_types: List[str],
    #     gold_answer: str, answers: list[str]
    # ):
    #     """
    #     We have a three phase rollout process:
    #     1. Generate question_expansion_factor clarification questions based on all previous history
    #     2. Generate answer_expansion_factor clarifying answers based on each clarification question and all previous history
    #     3. Generate inference answers based on all previous history 

    #     Q: Do I keep the dialogs as strings and tokenize every time or do I keep them as tokens and detokenize every time?
    #     A: I think I keep things as text cause it's what huggingface transformers wants
    #     """
    #     num_dialogs = len(images)

    #     # To simplify batching in this MVP, we need to have the number of dialogs to rollout be a multiple of the batch size
    #     assert num_dialogs % self.batch_size == 0, "Number of dialogs to rollout must be a multiple of the batch size"

    #     # We also do sanity checks on the other inputs
    #     assert num_dialogs == len(blurred_questions) == len(clear_questions) == len(gold_clarification_questions) == len(captions) == len(ambiguity_types) == len(answers), "All inputs must have the same number of dialogs"

    #     dialogs = []  # List of DialogTree objects
    #     active_dialog_nodes = []  # Tuples of (dialog_idx, node_idx)
    #     for dialog_idx in range(num_dialogs):
    #         init_question = blurred_questions[dialog_idx]
    #         init_image = images[dialog_idx]
    #         gold_answer = answers[dialog_idx]
    #         answers = answers[dialog_idx]
    #         dialogs.append(DialogTree(init_image, init_question, gold_answer, answers))
    #         active_dialog_nodes.append((dialog_idx, 0))
        
    #     # Now we rollout the dialog trees
    #     phase = NodeType.CLARIFICATION_QUESTION
    #     current_depth = 0

    #     while current_depth < self.cfg.dialog_tree.max_depth:
    #         # # Build the full history for each dialog
    #         # histories = []
    #         # for dialog_idx, node_idx in active_dialog_nodes:
    #         #     history, root_image = dialogs[dialog_idx].get_history_formatted(node_idx)
    #         #     histories.append((dialog_idx, node_idx,history, root_image))

    #         for batch in range(0, len(active_dialog_nodes), self.batch_size):
    #             batch_dialogs = active_dialog_nodes[batch:batch + self.batch_size]
    #             batch_histories = []
    #             for dialog_idx, node_idx in batch_dialogs:
    #                 history, root_image = dialogs[dialog_idx].get_history_formatted(node_idx)
    #                 batch_histories.append((dialog_idx, node_idx, history, root_image))

    #             if phase == NodeType.CLARIFICATION_QUESTION:
    #                 clarification_questions = self.clarification_model.generate(
                        
                


        