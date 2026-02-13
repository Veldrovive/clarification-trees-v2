"""
My test script for dialog tree generation.
Keeps the frontiers of all trees and expands.

We can use asyncio with vLLM to keep the logic for each tree separate and add queries to the queue for each model
independently. This greatly simplifies the logic for managing the frontiers of all trees and batching input.
"""

from clarification_trees.utils import add_inference_messages
from clarification_trees.dataset import ClearVQASample
import asyncio
from src.clarification_trees.models.vllm.remote_vllm_model import RemoteVLLMModel
import dotenv
dotenv.load_dotenv()

import hydra
from omegaconf import DictConfig
import numpy as np
import uuid
import random
from pathlib import Path
from tqdm import tqdm
import asyncio
from contextlib import asynccontextmanager

from clarification_trees.dialog_tree import DialogTree, NodeType, DialogTrajectory
from clarification_trees.models.vllm import QwenModelInputProcessor, CQModelWorker, GenericModelWorker, RemoteCQModel
from clarification_trees.models import Clusterer, construct_semantic_clusterer
from clarification_trees.dataset import ClearVQADataset
from clarification_trees.utils import set_seed, add_cq_messages, add_answer_messages

# def init_models(cfg: DictConfig):
#     lora_checkpoint_path = Path(cfg.paths.checkpoints.loras)

#     cq_model_cfg = OmegaConf.to_container(cfg['clarification_model'])
#     answer_model_cfg = OmegaConf.to_container(cfg['answer_model'])
#     clusterer_cfg = cfg['semantic_cluster_model']

#     cq_model_gpus = cfg.devices.clarification  # Like "3" or "2,3"
#     cq_model_num_gpus = cq_model_gpus.count(",") + 1
#     answer_model_gpus = cfg.devices.answer
#     answer_model_num_gpus = answer_model_gpus.count(",") + 1
#     clusterer_gpus = cfg.devices.semantic_cluster

#     print("Loading models...")
#     print(f"  Clarification model on {cq_model_num_gpus} GPUs ({cq_model_gpus})")
#     print(f"  Answer model on {answer_model_num_gpus} GPUs ({answer_model_gpus})")
#     print(f"  Clusterer on {clusterer_gpus}")
    
#     clusterer = construct_semantic_clusterer(clusterer_cfg, clusterer_gpus)

#     cq_model_processor = QwenModelInputProcessor(cq_model_cfg)
#     answer_model_processor = QwenModelInputProcessor(answer_model_cfg)

#     os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # Necessary to prevent Ray from overriding CUDA_VISIBLE_DEVICES to empty
    
#     cq_model = CQModelWorker.options( 
#         runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": cq_model_gpus}},
#         num_gpus=0  # 0 so that we do not get conflicts with vLLM trying to auto-assign GPUs
#     ).remote(cq_model_cfg, lora_checkpoint_path, n_gpus=cq_model_num_gpus)
    
#     answer_model = GenericModelWorker.options(
#         runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": answer_model_gpus}},
#         num_gpus=0  # 0 so that we do not get conflicts with vLLM trying to auto-assign GPUs
#     ).remote(answer_model_cfg, lora_checkpoint_path, n_gpus=answer_model_num_gpus)

#     cq_future = cq_model.ping.remote() 
#     ans_future = answer_model.ping.remote()

#     ray.get([cq_future, ans_future])
    
#     return clusterer, cq_model, answer_model, cq_model_processor, answer_model_processor

@asynccontextmanager
async def use_models(cfg: DictConfig):
    lora_checkpoint_path = Path(cfg.paths.checkpoints.loras)
    clarification_model_cfg = cfg.clarification_model
    answer_model_cfg = cfg.answer_model
    clarification_model_gpus = cfg.devices.clarification
    answer_model_gpus = cfg.devices.answer
    clarification_model_port = cfg.remote_vllm.clarification.port
    answer_model_port = cfg.remote_vllm.answer.port
    clarification_model_log_file = Path(cfg.remote_vllm.clarification.log_file)
    answer_model_log_file = Path(cfg.remote_vllm.answer.log_file)

    async def _start_vllm_server(model_cfg: DictConfig, gpus: list[int], port: int, log_file: Path):
        model = RemoteVLLMModel(
            model_cfg,
            lora_checkpoint_path,
            gpus=gpus,
            port=port,
            log_file=log_file
        )
        await model.initialize_server()
        return model

    clarification_model, answer_model = await asyncio.gather(
        _start_vllm_server(
            clarification_model_cfg,
            clarification_model_gpus,
            clarification_model_port,
            clarification_model_log_file
        ),
        _start_vllm_server(
            answer_model_cfg,
            answer_model_gpus,
            answer_model_port,
            answer_model_log_file
        )
    )

    clusterer_cfg = cfg['semantic_cluster_model']
    clusterer_gpus = cfg.devices.semantic_cluster
    clusterer = construct_semantic_clusterer(clusterer_cfg, clusterer_gpus)

    try:
        yield clarification_model, answer_model, clusterer
    finally:
        clarification_model.stop_server()
        answer_model.stop_server()

class DialogTreeDFSManager:
    def __init__(self, dialog_tree: DialogTree, max_depth: int):
        self.dialog_tree = dialog_tree
        self.max_depth = max_depth
        
        # Stack for DFS containing node indices. 
        # Initialize with the Root node (always index 0 in DialogTree init).
        self.stack = [DialogTree.ROOT]
        
        # Track depth of every node index to manage stopping criteria.
        # Root is at depth 0.
        self.node_depths = {DialogTree.ROOT: 0}

    def has_open_nodes(self) -> bool:
        return len(self.stack) > 0

    def get_next_node(self) -> tuple[DialogTrajectory, NodeType, int]:
        # DFS: Pop from the end (LIFO)
        node_id = self.stack.pop()
        
        # Retrieve node data from the tree.
        # DialogTree.nodes is a list of (parent_idx, DialogNode)
        parent_idx, node = self.dialog_tree.nodes[node_id]
        
        # Construct the trajectory context for this node
        trajectory = self.dialog_tree.get_trajectory(node_id)
        
        return trajectory, node.node_type, node_id

    def add_children(self, parent_node_id: int, new_node_texts: list[str], new_node_trans_probs: list[float], output_node_type: NodeType) -> list[int]:
        parent_depth = self.node_depths[parent_node_id]
        
        # Determine the depth of the new nodes.
        # Logic: 
        # ROOT (0) -> CQ (0) -> CA (1) -> CQ (1) -> CA (2)
        # Depth increases only when an Answer completes a pair.
        new_depth = parent_depth
        if output_node_type == NodeType.CLARIFYING_ANSWER:
            new_depth += 1
            
        # Iterate in reverse so that the first child (index 0) is pushed last 
        # and therefore popped first, maintaining order for the "first" cluster.
        new_node_ids = []
        for text, trans_prob in reversed(list(zip(new_node_texts, new_node_trans_probs))):
            # Add node to the data structure
            # Note: Generated nodes (CQ/CA) do not have images attached, so image=None.
            new_node_idx = self.dialog_tree.add_node(
                parent_idx=parent_node_id,
                node_type=output_node_type,
                image=None,
                response=text,
                transition_prob=trans_prob
            )
            
            # Record depth
            self.node_depths[new_node_idx] = new_depth
            
            # Decide if we should push this node to the stack for further expansion.
            # We stop expanding if we just completed a pair (Answer) and hit max_depth.
            # If we just added a Question, we always expand (to get the Answer).
            should_expand = True
            if output_node_type == NodeType.CLARIFYING_ANSWER and new_depth >= self.max_depth:
                should_expand = False
            
            if should_expand:
                self.stack.append(new_node_idx)
            
            new_node_ids.append(new_node_idx)
        
        return new_node_ids


# async def expand_tree(
#     cfg: DictConfig,
#     tree: DialogTree,
#     clusterer: Clusterer,
#     cq_model,
#     answer_model,
#     cq_processor: QwenModelInputProcessor,
#     answer_processor: QwenModelInputProcessor,
# ):
async def expand_tree(
    cfg: DictConfig,
    tree: DialogTree,
    clusterer: Clusterer,
    cq_model: RemoteVLLMModel,
    answer_model: RemoteVLLMModel,
):
    dialog_tree_config = cfg.dialog_tree
    max_depth = dialog_tree_config.max_depth
    question_expansion_factor = dialog_tree_config.question_expansion_factor
    answer_expansion_factor = dialog_tree_config.answer_expansion_factor
    question_diverse_sample_count = dialog_tree_config.question_diverse_sample_count
    answer_diverse_sample_count = dialog_tree_config.answer_diverse_sample_count
    inference_diverse_sample_count = dialog_tree_config.inference_diverse_sample_count

    dialog_tree_manager = DialogTreeDFSManager(tree, max_depth=max_depth)

    cq_node_types = set([NodeType.ROOT, NodeType.CLARIFYING_ANSWER])  # Node types that cause the tree to be expanded using the cq model
    answer_node_types = set([NodeType.CLARIFICATION_QUESTION])  # Node types that cause the tree to be expanded using the answer model

    async def _generate_inference(tree: DialogTree, answer_node_ids: list[int], n_outputs: int):
        for answer_node_id in answer_node_ids:
            dialog_trajectory = tree.get_trajectory(answer_node_id)
            messages = dialog_trajectory.to_messages(model_name="qwen-3-vl", use_img_path=True)
            add_inference_messages(messages, cfg=cfg)

            request_output = await answer_model.generate(messages, n_outputs=n_outputs, use_lora=False)
            generated_texts = [o.message.content for o in request_output.choices if o.message.content is not None]

            clusters, exemplars = clusterer.cluster(generated_texts)

            probabilities = [len(cluster) / len(generated_texts) for cluster in clusters]

            for exemplar, probability in zip(exemplars, probabilities):
                tree.add_node(
                    parent_idx=answer_node_id,
                    node_type=NodeType.INFERENCE,
                    response=exemplar,
                    transition_prob=probability
                )


    # We always start by making an inference from the root node.
    await _generate_inference(tree, [DialogTree.ROOT], inference_diverse_sample_count)
    while dialog_tree_manager.has_open_nodes():
        dialog_trajectory, input_node_type, node_id = dialog_tree_manager.get_next_node()
        messages = dialog_trajectory.to_messages(model_name="qwen-3-vl", use_img_path=True)

        if input_node_type in cq_node_types:
            add_cq_messages(messages, cfg=cfg)

            engine = cq_model
            sample_count = question_diverse_sample_count
            expansion_factor = question_expansion_factor
            output_node_type = NodeType.CLARIFICATION_QUESTION
            use_lora = True
        elif input_node_type in answer_node_types:
            assert tree.unambiguous_question is not None
            assert tree.answers is not None
            add_answer_messages(messages, unambiguous_question=tree.unambiguous_question, answers=tree.answers, cfg=cfg)

            engine = answer_model
            sample_count = answer_diverse_sample_count
            expansion_factor = answer_expansion_factor
            output_node_type = NodeType.CLARIFYING_ANSWER
            use_lora = False
        else:
            raise ValueError(f"Unknown node type: {input_node_type}")

        # request_output = await engine.generate.remote(vllm_inputs, n_outputs=sample_count, request_id=node_uuid)
        request_output = await engine.generate(messages, n_outputs=sample_count, use_lora=use_lora)
        generated_texts = [o.message.content for o in request_output.choices if o.message.content is not None]

        clusters, exemplars = clusterer.cluster(generated_texts)

        # We may have more clusters than the expansion factor allows
        # If this is the case, we randomly select a subset of the clusters to use
        # We may also have fewer in which case we use all of them
        if len(clusters) > expansion_factor:
            cluster_indices = random.sample(range(len(clusters)), expansion_factor)
            clusters = [clusters[i] for i in cluster_indices]
            exemplars = [exemplars[i] for i in cluster_indices]
        total_allowed_texts = sum([len(cluster) for cluster in clusters])
        probabilities = [len(cluster) / total_allowed_texts for cluster in clusters]
        assert np.isclose(sum(probabilities), 1.0)

        new_node_ids = dialog_tree_manager.add_children(
            parent_node_id=node_id,
            new_node_texts=exemplars,
            new_node_trans_probs=probabilities,
            output_node_type=output_node_type
        )

        if output_node_type == NodeType.CLARIFYING_ANSWER:
            await _generate_inference(tree, new_node_ids, answer_diverse_sample_count)

    return tree

async def process_dataset_lazily(
    cfg: DictConfig,
    dataset: ClearVQADataset,
    clusterer: Clusterer,
    cq_model: RemoteVLLMModel,
    answer_model: RemoteVLLMModel,
    N_parallel_trees: int = 10,
    out_dir: Path | None = None
):
    # Configuration
    total_items = len(dataset)
    
    # We keep a set of currently running tasks
    active_tasks = set()
    
    # Initialize progress bar
    pbar = tqdm(total=total_items, desc="Expanding Trees")

    for i in range(total_items):
        # 1. THROTTLING: If we are full, wait for at least one task to finish
        if len(active_tasks) >= N_parallel_trees:
            # wait returns two sets: 'done' tasks and 'pending' tasks
            done, pending = await asyncio.wait(
                active_tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 2. CLEANUP: Process finished tasks and remove from active set
            for task in done:
                try:
                    finished_tree: DialogTree = await task # Retrieve the result (or raise exception)
                    if out_dir:
                        img_path = finished_tree.init_image_path
                        if img_path is None:
                            out_file = out_dir / f"tree_{uuid.uuid4()}.json"
                        else:
                            img_name = img_path.stem
                            out_file = out_dir / f"tree_{img_name}_{uuid.uuid4()}.json"
                        finished_tree.save(out_file)
                    else:
                        print("Warning: No output directory provided. Tree will not be saved.")
                except Exception as e:
                    print(f"Task failed: {e}")
                finally:
                    pbar.update(1)
            
            # Update active_tasks to only contain the ones still running
            active_tasks = pending

        # 3. LAZY LOADING: Now that we have a slot, load the data
        # The image is loaded into memory HERE, not before.
        sample = dataset[i] 
        tree = DialogTree(
            init_question=sample.blurred_question,
            init_image=None,
            init_image_path=sample.image_path,
            init_image_caption=sample.caption,
            unambiguous_question=sample.question,
            gold_answer=sample.gold_answer,
            answers=sample.answers
        )

        # 4. DISPATCH: Create the coroutine and track it
        # We wrap it in a task immediately
        task = asyncio.create_task(
            expand_tree(
                cfg=cfg,
                tree=tree,
                clusterer=clusterer,
                cq_model=cq_model,
                answer_model=answer_model,
            )
        )
        active_tasks.add(task)

    # 5. DRAIN: Wait for the final batch of tasks to finish after the loop
    if active_tasks:
        done, _ = await asyncio.wait(active_tasks)
        for task in done:
            try:
                finished_tree: DialogTree = await task
                if out_dir:
                    img_path = finished_tree.init_image_path
                    if img_path is None:
                        out_file = out_dir / f"tree_{uuid.uuid4()}.json"
                    else:
                        img_name = img_path.stem
                        out_file = out_dir / f"tree_{img_name}_{uuid.uuid4()}.json"
                    finished_tree.save(out_file)
            except Exception as e:
                print(f"Task failed: {e}")
            pbar.update(1)
            
    pbar.close()


async def run_single_tree_test(cfg: DictConfig, sample: ClearVQASample):
    test_tree = DialogTree(
        init_question=sample.blurred_question,
        init_image=None,
        init_image_path=sample.image_path,
        init_image_caption=sample.caption,
        unambiguous_question=sample.question,
        gold_answer=sample.gold_answer,
        answers=sample.answers
    )

    async with use_models(cfg) as (cq_model, answer_model, clusterer):
        await expand_tree(
            cfg=cfg,
            tree=test_tree,
            clusterer=clusterer,
            cq_model=cq_model,
            answer_model=answer_model,
        )

async def run_expand_trees(cfg: DictConfig, ds: ClearVQADataset, out_dir: Path, n_parallel_trees: int = 10):
    async with use_models(cfg) as (cq_model, answer_model, clusterer):
        await process_dataset_lazily(
            cfg=cfg,
            dataset=ds,
            clusterer=clusterer,
            cq_model=cq_model,
            answer_model=answer_model,
            N_parallel_trees=n_parallel_trees,
            out_dir=out_dir
        )


@hydra.main(config_path="src/clarification_trees/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    SINGLE_TREE_TEST = False
    print(cfg)
    set_seed(cfg.seed)

    out_path = Path("./data/trees")
    out_path.mkdir(parents=True, exist_ok=True)

    ds = ClearVQADataset(load_images=False)

    if SINGLE_TREE_TEST:
        sample = ds[0]
        asyncio.run(run_single_tree_test(cfg, sample))
    else:
        N_parallel_trees = 10

        # from torch.utils.data import Subset
        # ds = Subset(ds, range(500))

        asyncio.run(run_expand_trees(cfg, ds, out_path, n_parallel_trees=N_parallel_trees))

    # # Construct tree roots using the train set of ClearVQA
    # ds = ClearVQADataset(load_images=False)
    # test_tree = DialogTree(
    #     init_question=ds[0][1]["blurred_question"],
    #     init_image=ds[0][0],
    #     init_image_path=
    #     unambiguous_question=ds[0][1]["question"],
    #     gold_answer=ds[0][1]["gold_answer"],
    #     answers=ds[0][1]["answers"]
    # )

    # from torch.utils.data import Subset
    # ds = Subset(ds, range(500))
    

    # # clusterer, cq_model, answer_model, cq_model_processor, answer_model_processor = init_models(cfg)
    # clusterer, remote_cq_model = init_models(cfg)

    # # asyncio.run(
    # #     expand_tree(
    # #         cfg=cfg,
    # #         tree=test_tree,
    # #         clusterer=clusterer,
    # #         cq_model=cq_model,
    # #         answer_model=answer_model,
    # #         cq_processor=cq_model_processor,
    # #         answer_processor=answer_model_processor
    # #     )
    # # )

    # N_parallel_trees = 100
    # asyncio.run(
    #     # process_dataset_lazily(
    #     #     cfg,
    #     #     ds,
    #     #     clusterer,
    #     #     cq_model,
    #     #     answer_model,
    #     #     cq_model_processor,
    #     #     answer_model_processor,
    #     #     N_parallel_trees,
    #     #     out_dir=out_path
    #     # )
    #     process_dataset_lazily(
    #         cfg,
    #         ds,
    #         clusterer,
    #         remote_cq_model,
    #         N_parallel_trees,
    #         out_dir=out_path
    #     )
    # )
    


if __name__ == "__main__":
    main()