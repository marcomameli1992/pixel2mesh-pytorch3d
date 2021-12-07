"""
The main file with the basic configuration
"""
from train import train
import json
import argparse

from torch.utils.data import DataLoader

from network.basenetwork.convolutionalnetwork import Convolutional
from network.basenetwork.graphconvolutionalnetwork import GraphConvolutional

from data.datasetmanagement.shapenet import ShapeNetDataset

# argument parser preparation
parser = argparse.ArgumentParser()
parser.add_argument('--configuration_file', '-cf', required=True, help="Represent the path for configuration file.")
args = parser.parse_args()

if __name__ == "__main__":

    with open(args.configuration_file, 'r') as config_file:
        config_parameter = json.load(config_file)

    # Model creation
    img_convolutional = Convolutional(input_channel=4, pooling=config_parameter['convolutional_network']['pooling'], pooling_type=config_parameter['convolutional_network']['pooling_type'])
    gph_convolutional = GraphConvolutional(first_block_input_dim=64, second_block_input_dim=128, third_block_input_dim=256, hidden_dim=256, output_dim=3)

    # Dataset init
    train_dataset = ShapeNetDataset(dataset_file_path=config_parameter['data_file_list'], img_dataset_path=config_parameter['img_base_path'], model_dataset_path=config_parameter['obj_base_path'], img_name=config_parameter['img_name'], split='train')
    validation_dataset = ShapeNetDataset(dataset_file_path=config_parameter['data_file_list'], img_dataset_path=config_parameter['img_base_path'], model_dataset_path=config_parameter['obj_base_path'], img_name=config_parameter['img_name'], split='validation')

    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config_parameter['training']['batch_size'])
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=config_parameter['training']['batch_size'])

    train(config_parameter, img_convolutional, gph_convolutional, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader)
