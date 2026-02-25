from clarification_trees.dialog_tree import NodeType
from omegaconf import DictConfig
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from typing import TypedDict

from clarification_trees.dialog_tree import DialogTree, TreeSidecar

DEFAULT_CLEARVQA_DATA_PATH = Path(__file__).parents[2] / "data" / "clearvqa"
DEFAULT_TREES_DATA_PATH = Path(__file__).parents[2] / "data" / "trees"

@dataclass
class ClearVQASample:
    question_id: str
    question: str
    gold_answer: str
    image: Image.Image | None
    answers: list[str]
    dataset: str
    caption: str
    blurred_question: str
    clarification_question: str
    prompt_type: str
    image_path: Path
    ambiguity_category: str | None = None
    

class ClearVQADataset(Dataset):
    def __init__(self,
        data_path: Path = DEFAULT_CLEARVQA_DATA_PATH,
        table_name = "train_annotated.jsonl",
        transform = None,
        load_images: bool = True
    ):
        self.transform = transform
        self.load_images = load_images

        self.data_path = data_path
        self.images_path = self.data_path / "images"
        assert self.images_path.is_dir(), "Images directory does not exist"
        self.table_path = self.data_path / table_name
        assert self.table_path.exists(), f"Table file {self.table_path} does not exist"
        self.load_tables()

        print(f"Train headers: {self.table_df.columns}")

    def load_tables(self):
        print("Loading table...")
        self.table_df = pd.read_json(self.table_path, lines=True)
        print("Finished loading tables")

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, index) -> ClearVQASample:
        sample = self.table_df.iloc[index].to_dict()

        # The loaded sample has the image as a string (the name of the image)
        # We use this to compute the image path and replace it with the actual image object if we are loading images
        image_name = sample['image']
        image_path = self.images_path / image_name
        assert image_path.exists(), f"Image not found at {image_path}"
        sample['image_path'] = image_path
        sample['image'] = None
        
        if self.load_images:
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    sample['image'] = img
                except FileNotFoundError:
                    raise FileNotFoundError(f"Image not found at {image_path}")
            else:
                raise FileNotFoundError(f"Image not found at {image_path}")
        
        return ClearVQASample(**sample)


@dataclass
class ClarificationTreeSample:
    tree: DialogTree
    parent_node_idx: int
    child_node_idxs: list[int]
    advantages: list[float]
    logits: list[torch.Tensor] | None

class ClarificationTreeSampleDict(TypedDict):
    tree_idx: int
    parent_node_idx: int
    child_node_idxs: list[int]

class ClarificationTreeDataset(Dataset):
    trees: list[DialogTree]
    sidecars: list[TreeSidecar]
    samples: list[ClarificationTreeSampleDict]
    cached_reward_tree_idxs: set[int]
    
    def __init__(self,
        cfg: DictConfig,
        trees_path: Path = DEFAULT_TREES_DATA_PATH,
        transform = None,
        load_images: bool = True,
        precompute_rewards: bool = True
    ):
        self.cfg = cfg
        self.trees_path = trees_path
        self.transform = transform
        self.load_images = load_images
        self.precompute_rewards = precompute_rewards

        self._load_trees()

    def _load_trees(self):
        trees = []
        sidecars = []
        cached_reward_tree_idxs = set()

        samples = []  # [{"tree_idx": int, "parent_node_idx": int, "child_node_idxs": list[int]}]

        tree_dirs = list(self.trees_path.iterdir())
        for tree_dir in tqdm(tree_dirs, desc="Loading trees"):
            if not tree_dir.is_dir():
                print(f"Skipping non-directory {tree_dir}")
                continue
            
            tree_path = tree_dir / "tree.json"
            if not tree_path.exists():
                print(f"Skipping non-tree {tree_path}")
                continue
            
            sidecar_path = tree_dir / "tree_sidecar.json"
            if not sidecar_path.exists():
                print(f"Skipping non-sidecar {sidecar_path}")
                continue
            
            tree = DialogTree.load(tree_path)
            sidecar = TreeSidecar.load(sidecar_path, self.cfg)
            
            tree_idx = len(trees)
            trees.append(tree)
            sidecars.append(sidecar)
            if self.precompute_rewards:
                sidecar.compute_rewards()  # Caches the rewards and advantages
                cached_reward_tree_idxs.add(tree_idx)

            for parent_node_idx, parent_node in tree.get_nodes():
                child_cq_idxs = tree.get_children_idxs(parent_node_idx, type_filter=NodeType.CLARIFICATION_QUESTION)
                if len(child_cq_idxs) == 0:
                    continue
                
                samples.append(ClarificationTreeSampleDict(
                    tree_idx=tree_idx,
                    parent_node_idx=parent_node_idx,
                    child_node_idxs=child_cq_idxs
                ))

        self.trees: list[DialogTree] = trees
        self.sidecars: list[TreeSidecar] = sidecars
        self.samples: list[ClarificationTreeSampleDict] = samples
        self.cached_reward_tree_idxs: set[int] = cached_reward_tree_idxs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> ClarificationTreeSample:
        sample = self.samples[index]
        
        tree_index = sample["tree_idx"]
        parent_node_idx = sample["parent_node_idx"]
        child_node_idxs = sample["child_node_idxs"]

        if tree_index not in self.cached_reward_tree_idxs:
            self.sidecars[tree_index].compute_rewards()
            self.cached_reward_tree_idxs.add(tree_index)

        tree = self.trees[tree_index]
        sidecar = self.sidecars[tree_index]

        child_advantages = [sidecar.advantage_cache[child_node_idx] for child_node_idx in child_node_idxs]

        return ClarificationTreeSample(
            tree=tree,
            parent_node_idx=parent_node_idx,
            child_node_idxs=child_node_idxs,
            advantages=child_advantages,
            logits=None
        )


if __name__ == "__main__":
    # ds = ClearVQADataset(load_images=True)
    # print(ds[0])

    import json
    ds = ClarificationTreeDataset(cfg=DictConfig({}), precompute_rewards=False)
    print(f"Dataset size: {len(ds)}")
    sample = ds[1000]

    traj = sample.tree.get_trajectory(sample.parent_node_idx)
    messages = traj.to_messages("qwen-3-vl", use_img_path=True)
    print(json.dumps(messages, indent=2))

    for i, child_node_idx in enumerate(sample.child_node_idxs):
        child_node = sample.tree.get_node(child_node_idx)
        print(f"  (Adv: {sample.advantages[i]}) {child_node}")