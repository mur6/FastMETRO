import torch
from torch.utils.data import Dataset, Sampler


def main_1():
    # Define the custom dataset
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Define the custom sampler
    class CustomSampler(Sampler):
        def __init__(self, data_source):
            self.data_source = data_source

        def __iter__(self):
            def _iter():
                count = 0
                for i in range(len(self.data_source)):
                    # print(i)
                    if i % 3 == 0:
                        yield i
                        # count += 1

            return _iter()

        def __len__(self):
            return 1  # len(self.data_source)

    # Create the dataset and sampler
    data = [7, 1, 2, 8, 1, 2, 21]
    dataset = CustomDataset(data)
    sampler = CustomSampler(dataset)

    # Use the custom dataset and sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)

    for i, batch in enumerate(dataloader):
        print(i, batch)


import torch
from torch.utils.data import Dataset, ConcatDataset


# Define the first custom dataset
class CustomDataset1(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Define the second custom dataset
class CustomDataset2(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create the first dataset
data1 = [1, 2, 3, 4, 5]
dataset1 = CustomDataset1(data1)

# Create the second dataset
data2 = [6, 7, 8, 9, 10]
dataset2 = CustomDataset2(data2)

# Append the two datasets
dataset = ConcatDataset([dataset1, dataset2])
for k in dataset:
    print(k)
