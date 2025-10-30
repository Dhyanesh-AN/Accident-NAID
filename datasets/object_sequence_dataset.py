
import torch
from torch.utils.data import Dataset
import pandas as pd
import ast

class ObjectSequenceDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.drop(columns=['Unnamed: 0', 'video_id', 'frame'], errors='ignore')
        assert 'target' in self.df.columns, f"'target' column missing in {csv_path}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        obj_data = row.drop('target').values
        parsed = []
        for s in obj_data:
            coords = ast.literal_eval(s) if isinstance(s, str) else [0, 0, 0, 0]
            parsed.append(coords)
        tensor_data = torch.tensor(parsed, dtype=torch.float32).view(30, 10, 4)
        target = torch.tensor(row['target'], dtype=torch.float32)
        return tensor_data.unsqueeze(0), target.unsqueeze(0)  # (1, 30, 10, 4), (1,)
