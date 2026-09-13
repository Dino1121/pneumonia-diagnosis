from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split,
        transform=None,
        project_root=None
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform

        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)

        df = pd.read_csv(self.csv_path)

        # train / val / test 중 원하는 split만 선택
        self.data = df[df["split"] == split].reset_index(drop=True)

        self.label_map = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = Path(row["filepath"])

        if not image_path.is_absolute():
            image_path = self.project_root / image_path

        image = Image.open(image_path)

        # NORMAL → 0, PNEUMONIA → 1
        label = self.label_map[row["label"]]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label