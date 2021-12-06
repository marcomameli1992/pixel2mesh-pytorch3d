"""
Dataset Management for the shapenet dataset
:author: Marco Mameli
:return: Dataset type
:rtype: torch.dataset.Dataset
"""
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from pytorch3d.structures.meshes import join_meshes_as_batch
from pytorch3d.io import load_objs_as_meshes, load_obj
from skimage.io import imread
import os

class ShapeNetDataset(Dataset):
    """
    Dataset class for the dataset
    :author: Marco Mameli

    """

    def __init__(self, dataset_file_path: str, img_dataset_path: str, model_dataset_path: str, img_name: str, split: str, load_textures: bool = False):
        #% Init the dataset with the generation of the file list and object list

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.img_dict = {}
        with open(dataset_file_path + split + '.txt', 'r') as d_t:
            for line in d_t.readlines():
                self.img_dict[line.split(',')[0].strip()] = line.split(',')[1].strip()

        self.obj_list = []
        delete = []

        # Usare per aprire le mesh per la seconda parte
        for i, s in enumerate(self.img_dict.keys()):
            if os.path.exists(s.replace(img_dataset_path, model_dataset_path).replace('rendering', 'models').replace(img_name + '.png', 'model_normalized.obj').strip()):
                self.obj_list.append(s.replace(img_dataset_path, model_dataset_path).replace('rendering', 'models').replace(img_name + '.png', 'model_normalized.obj'))
            else:
                delete.append(s)

        self.img_list = [f for f in self.img_dict.keys() if f not in delete]
        #self.obj_list = [s.replace(image_dataset_path, model_dataset_path).replace('rendering', 'models').replace(img_name + '.png', 'model_normalized.obj') for s in self.image_list]

    def __getitem__(self, item) ->Tuple:
        #% Get the item from dataset: Image and Obj

        image = imread(self.img_list[item].strip())
        image = self.transform(image)

        data = {}
        data['image'] = image
        data['img_path'] = self.img_list[item].strip()
        data['class'] = self.img_dict[self.img_list[item].strip()]

        return data

    def __len__(self) -> int:
        return len(self.img_list)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = ShapeNetDataset(dataset_file_path='/mnt/e/Progetti/ComputerGraphics/LandscapeGeneration-Vienna/MeshNetwork/pixel2mesh/data/dataloader/data_list/', img_dataset_path="/mnt/e/Progetti/ComputerGraphics/LandscapeGeneration-Vienna/MeshNetwork/Dataset/ShapeNet/ShapeNetP2M", model_dataset_path="/mnt/e/Progetti/ComputerGraphics/LandscapeGeneration-Vienna/MeshNetwork/Dataset/ShapeNet/ShapeNetCorev2", img_name='00', split='test')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    img = next(iter(dataloader))
    print(img['image'].shape,img['img_path'], img['class']) # the dictionary has the image element has tensor of image tensor and the img_path as list of string