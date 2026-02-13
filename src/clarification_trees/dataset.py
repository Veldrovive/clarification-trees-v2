import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

DEFAULT_DATA_PATH = Path(__file__).parents[2] / "data" / "clearvqa"

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
    

class ClearVQADataset(Dataset):
    def __init__(self,
        data_path: Path = DEFAULT_DATA_PATH,
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


if __name__ == "__main__":
    ds = ClearVQADataset(load_images=True)
    print(ds[0])