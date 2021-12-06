"""
Training function for the network configuration
"""
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from pytorch3d import loss as l3d
from torch import cuda
from pytorch3d.io import load_objs_as_meshes

#TODO Adding the possibility to pre-train the convolutional network

def train(config, convolutional_model: nn.Module, graph_model: nn.Module, train_data: DataLoader, validation_data: DataLoader, epochs: int, running_save):
    # optmizer init
    image_optimizer = optim.Adam(convolutional_model.parameters(), lr=0.001) # See weight decay
    mesh_optimizare = optim.Adam(graph_model.parameters(), lr=0.001)

    # model on GPU
    device = "cuda:0" if cuda.is_available() else "cpu"
    convolutional_model.to(device)
    graph_model.to(device)

    # define the optimizer
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data):
            #
            image_optimizer.zero_grad()
            mesh_optimizare.zero_grad()
            
            # forward + backward + optimize
            conv16, conv32, conv64, conv128, conv256, conv512 = convolutional_model(data['image'])

            mesh_loss_on_batch = 0.0
            if len(conv16.shape()) == 4 or len(conv32.shape()) == 4 or len(conv64.shape()) == 4 or len(conv128.shape()) == 4 or len(conv256.shape()) == 4 or len(conv512.shape()) == 4:
                # Batch size different from 1
                for i in conv64.shape()[0]:
                    # opening mesh for the image in the batch
                    mesh_path = data['img_path'].replace(config['img_base_path'], config['obj_base_path']).replace(
                        '/rendering/' + config['img_name'] + '.png', '/models/' + config['obj_name'] + '.obj')
                    mesh = load_objs_as_meshes([mesh_path])
                    mesh1, mesh2, mesh3 = graph_model(mesh, conv64[i].unsqueeze(0), conv128[i].unsqueeze(0), conv256[i].unsqueeze(0)) # TODO Define the data input

                    mesh_loss_on_batch += l3d.mesh_edge_loss([mesh, mesh1]) + l3d.mesh_laplacian_smoothing([mesh, mesh1]) + l3d.mesh_normal_consistency([mesh, mesh1])
                mesh_loss_on_batch = mesh_loss_on_batch / conv64.shape()[0]
            else:
                raise NotImplementedError("At the moment of the code creation the batch is not supported by pytorch3d v 0.5.0.\n" +
                                          "Please contact the developer at mameli.1.marco@gmail.com with object PyTorch3D Pixel2Mesh Re-Implementation" +
                                          "and asck for code updates")
            # backpropagation # TODO understand the backpropagation see https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
            mesh_loss_on_batch