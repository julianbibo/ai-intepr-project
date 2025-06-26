from torch.utils.data import Dataset
import torch
from pathlib import Path


class ActivationsDataset(Dataset):
    def __init__(self, data_dir, instruments, seeds, split: str, layer: int):
        self.data_dir = Path(data_dir)
        self.instruments = instruments
        self.seeds = seeds
        self.split = split
        self.layer = layer

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of ['train', 'val', 'test'], got {split}")

        X_list = []
        Y_list = []
        self.prompts = []

        for instrument in instruments:
            for seed in seeds:
                file_path = self.data_dir / instrument / \
                    f"seed_{seed}" / split / f"layer_{layer:02d}.pt"
                if not file_path.exists():
                    raise FileNotFoundError(f"Data file not found: {file_path}")
                data = torch.load(file_path, map_location="cpu")
                X_list.append(data["X"])
                Y_list.append(data["Y"])
                self.prompts.extend(data["prompts"])

        self.X = torch.cat(X_list, dim=0)
        self.Y = torch.cat(Y_list, dim=0)
        print(f"{self.X.shape=}, {self.Y.shape=}, {len(self.prompts)=}")

        assert self.X.size(0) == self.Y.size(0) == len(self.prompts), \
            "Mismatch in lengths of X, Y, and prompts"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.prompts[idx]
