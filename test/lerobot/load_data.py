import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = "/home/ryu-yang/DoRobot/dataset/20251223/user/Fold clothes_Fold松灵分体-8号_869_81321"

# 1) Load from the Hub (cached locally)
dataset = LeRobotDataset(repo_id="Fold clothes_Fold松灵分体-8号_869_81321", root=root)

# 2) Random access by index
sample = dataset[100]
print(sample)


batch_size = 16
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
