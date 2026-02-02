import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).parents[2] / "data" / "clearvqa"

class ClearVQADataset(Dataset):
    def __init__(self,
        data_path: Path = DEFAULT_DATA_PATH,
        table_name = "train_annotated.jsonl",
        transform = None
    ):
        self.transform = transform

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

    def __getitem__(self, index):
        sample = self.table_df.iloc[index].to_dict()
        image_name = sample['image']
        image_path = self.images_path / image_name
        sample['image_path'] = image_path
        
        if image_path.exists():
            try:
                img = Image.open(image_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found at {image_path}")
        else:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        return img, sample


if __name__ == "__main__":
    ds = ClearVQADataset()
    print(ds[0])