"""
Dataset Management for the shapenet dataset
:author: Marco Mameli
:return: Dataset type
:rtype: torch.dataset.Dataset
"""
from torch.utils.data import Dataset
from glob import glob

class ShapeNetDataset(Dataset):
    """
    Dataset class for the dataset
    :author: Marco Mameli

    """

    def __init__(self, dataset_path: str):
        #% Init the dataset with the generation of the file list and object list
        self.image_list = glob(dataset_path)
        self.obj_list = glob(dataset_path)


    def __getitem__(self, item):
        return item

    def __len__(self):
        return 126