"""
data structures:
data = torch.load("residual_data/guitar/layer_05_last.pt")
{
    "X": intermediate residuals  (N, D) 
    "Y": final layer residuals   (N, D) 
    "prompts": [prompt_1, ..., prompt_N],
    "layer": 05,
    "instrument": "guitar"
}

AND

data = torch.load("residual_data/guitar/layer_05_all.pt")
{
    "X": intermediate residuals  (N, T, D) 
    "Y": final layer residuals   (N, T, D) 
    "prompts": [prompt_1, ..., prompt_N],
    "layer": 05,
    "instrument": "guitar"
}
"""

import torch
print(f"info seeds-1-2-3")
data = torch.load("combined_data_seeds-1-2-3/trumpet/layer_00_train.pt")
print(data["X"].shape, data["Y"].shape)
print(data["layer"], data["instrument"])
print(data["X"][0], data["Y"][0])

data = torch.load("combined_data_seeds-1-2-3/trumpet/layer_00_val.pt")
print(data["X"].shape, data["Y"].shape)
print(data["layer"], data["instrument"])
print(data["X"][0], data["Y"][0])

data = torch.load("combined_data_seeds-1-2-3/trumpet/layer_00_test.pt")
print(data["X"].shape, data["Y"].shape)
print(data["layer"], data["instrument"])
print(data["X"][0], data["Y"][0])




print("\n\ninfo seed-1")
data = torch.load("combined_data_seed-1/trumpet/layer_00_train.pt")
print(data["X"].shape, data["Y"].shape)
print(data["layer"], data["instrument"])
print(data["X"][0], data["Y"][0])

data = torch.load("combined_data_seed-1/trumpet/layer_00_val.pt")
print(data["X"].shape, data["Y"].shape)
print(data["layer"], data["instrument"])
print(data["X"][0], data["Y"][0])

data = torch.load("combined_data_seed-1/trumpet/layer_00_test.pt")
print(data["X"].shape, data["Y"].shape)
print(data["layer"], data["instrument"])
print(data["X"][0], data["Y"][0])


# print(data["prompts"])


# data = torch.load("residual_data/violin/seed_1/layer_00_all.pt")
# print(data["X"].shape, data["Y"].shape)
# print(data["layer"], data["instrument"])
# print(data["prompts"])


# """

# Needs to change based on how exactly data will be saved in generate_train_data.py
# """


# def load_residual_data(path: str):
#     data = torch.load(path)
#     print(f"Loaded: {path}")
#     print(f"X shape: {data['X'].shape}, Y shape: {data['Y'].shape}")
#     return data["X"], data["Y"], data["prompts"], data["intermediate_layer"], data["final_layer"]


# X_last, Y_last, prompts, il, fl = load_residual_data("residual_data/residuals_last_token.pt")



# ##########################
# # OR
# ##########################


# from torch.utils.data import Dataset

# class ResidualTranslationDataset(Dataset):
#     def __init__(self, X: torch.Tensor, Y: torch.Tensor):
#         assert X.shape[0] == Y.shape[0]
#         self.X = X
#         self.Y = Y

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx]


# dataset = ResidualTranslationDataset(X_last, Y_last)
# x, y = dataset[0]
