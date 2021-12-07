"""
Training function for the network configuration
:author: Marco Mameli

"""
import torch
import wandb
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from pytorch3d import loss as l3d
from torch import cuda
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
import neptune.new as neptune
import numpy as np
from validation import validation

#TODO Adding the possibility to pre-train the convolutional network

def train(config, convolutional_model: nn.Module, graph_model: nn.Module, train_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:
    """
    The training function for the training of th network
    Args:
        config: the configuration structure from the JSON
        convolutional_model: the convolutional model
        graph_model: the graph convolutional model
        train_dataloader: the training data loader
        validation_dataloader: the validation data loader
    """

    #% Initialize experiment history.
    # Check for neptune
    run = None
    if config['neptune']['activate']:
        run = neptune.init(
            project=config['neptune']['project'],
            api_token=config['neptune']['api_token']
        )
    elif config['w&b']['activate']:
        wandb.init(project=config['w&b']['project'], entity=config['w&b']['entity'])

    # Number of parameters computation
    cnn_parameters = filter(lambda p: p.requires_grad, convolutional_model.parameters())
    cnn_params = sum([np.prod(p.size()) for p in cnn_parameters])
    gcn_parameters = filter(lambda p: p.requires_grad, graph_model.parameters())
    gcn_params = sum([np.prod(p.size()) for p in gcn_parameters])
    params = {
        'learning_rate': config['training']['learning_rate'],
        'optimizer': config['training']['optimizer'],
        'gpu': torch.cuda.get_device_name(),
        'convolutional_model_n_parameters': cnn_params,
        'graph_model_n_parameters': gcn_params
    }
    if config['neptune']['activate']:
        run["parameters"] = params
    elif config['w&b']['activate']:
        wandb.config(
            params
        )
    # optmizer init
    if config['training']['optimizer'] == "sgd":
        image_optimizer = optim.SGD(convolutional_model.parameters(), lr=config['training']['learning_rate'], momentum=config['training']['momentum'])
        mesh_optimizare = optim.SGD(graph_model.parameters(), lr=config['training']['learning_rate'], momentum=config['training']['momentum'])
    if config['training']['optimizer'] == "adam":
        image_optimizer = optim.Adam(convolutional_model.parameters(), lr=config['training']['learning_rate'])
        mesh_optimizare = optim.Adam(graph_model.parameters(), lr=config['training']['learning_rate'])

    # model on GPU
    device = "cuda:0" if cuda.is_available() else "cpu"
    convolutional_model.to(device)
    graph_model.to(device)

    # Create the subdivision for label mesh
    subdivide = SubdivideMeshes()

    # define the optimizer
    for epoch in range(config['training']['epochs']):
        for i, data in enumerate(train_dataloader):
            #
            image_optimizer.zero_grad()
            mesh_optimizare.zero_grad()
            
            # forward + backward + optimize
            conv16, conv32, conv64, conv128, conv256, conv512 = convolutional_model(data['image'])

            mesh1_loss_on_batch = 0.0
            mesh2_loss_on_batch = 0.0
            mesh3_loss_on_batch = 0.0
            if len(conv16.shape()) == 4 or len(conv32.shape()) == 4 or len(conv64.shape()) == 4 or \
                    len(conv128.shape()) == 4 or len(conv256.shape()) == 4 or len(conv512.shape()) == 4:
                # Batch size different from 1
                for i in conv64.shape()[0]:
                    # opening mesh for the image in the batch
                    label_mesh_path = data['img_path'].replace(config['img_base_path'], config['obj_base_path']).replace(
                        '/rendering/' + config['img_name'] + '.png', '/models/' + config['obj_name'] + '.obj')

                    label_mesh1 = load_objs_as_meshes([label_mesh_path], device=device)
                    label_mesh2 = subdivide(label_mesh1)
                    label_mesh3 = subdivide(label_mesh2)

                    mesh = ico_sphere(level=0, device=device)

                    mesh1, mesh2, mesh3 = graph_model(mesh, conv64[i].unsqueeze(0), conv128[i].unsqueeze(0), conv256[i].unsqueeze(0))

                    # point cloud conversion
                    point_label_mesh1 = sample_points_from_meshes(label_mesh1, num_samples=5000)
                    point_label_mesh2 = sample_points_from_meshes(label_mesh2, num_samples=5000)
                    point_label_mesh3 = sample_points_from_meshes(label_mesh3, num_samples=5000)

                    point_mesh1 = sample_points_from_meshes(mesh1, num_samples=5000)
                    point_mesh2 = sample_points_from_meshes(mesh2, num_samples=5000)
                    point_mesh3 = sample_points_from_meshes(mesh3, num_samples=5000)

                    # chamfer distance
                    chamfer1 = l3d.chamfer_distance(point_mesh1, point_label_mesh1)
                    chamfer2 = l3d.chamfer_distance(point_mesh2, point_label_mesh2)
                    chamfer3 = l3d.chamfer_distance(point_mesh3, point_label_mesh3)
                    mesh1_loss_on_batch += chamfer1 * config['loss_weights']['chamfer']
                    mesh2_loss_on_batch += chamfer2 * config['loss_weights']['chamfer']
                    mesh3_loss_on_batch += chamfer3 * config['loss_weights']['chamfer']

                    # edge smooth
                    edge1 = l3d.mesh_edge_loss(mesh1)
                    edge2 = l3d.mesh_edge_loss(mesh2)
                    edge3 = l3d.mesh_edge_loss(mesh3)
                    mesh1_loss_on_batch += edge1 * config['loss_weights']['edge']
                    mesh2_loss_on_batch += edge2 * config['loss_weights']['edge']
                    mesh3_loss_on_batch += edge3 * config['loss_weights']['edge']

                    # laplacian smooth
                    laplacian1 = l3d.mesh_laplacian_smoothing(mesh1)
                    laplacian2 = l3d.mesh_laplacian_smoothing(mesh2)
                    laplacian3 = l3d.mesh_laplacian_smoothing(mesh3)
                    mesh1_loss_on_batch += laplacian1 * config['loss_weights']['laplacian']
                    mesh2_loss_on_batch += laplacian2 * config['loss_weights']['laplacian']
                    mesh3_loss_on_batch += laplacian3 * config['loss_weights']['laplacian']

                    # normal convalidation
                    normal1 = l3d.mesh_normal_consistency(mesh1)
                    normal2 = l3d.mesh_normal_consistency(mesh2)
                    normal3 = l3d.mesh_normal_consistency(mesh3)
                    mesh1_loss_on_batch += normal1 * config['loss_weights']['normal']
                    mesh2_loss_on_batch += normal2 * config['loss_weights']['normal']
                    mesh3_loss_on_batch += normal3 * config['loss_weights']['normal']

                    loss = {
                        'chamfer_distance_1': chamfer1,
                        'chamfer_distance_2': chamfer2,
                        'chamfer_distance_3': chamfer3,
                        'edge_loss_1': edge1,
                        'edge_loss_2': edge2,
                        'edge_loss_3': edge3,
                        'laplacia_loss_1': laplacian1,
                        'laplacia_loss_2': laplacian2,
                        'laplacia_loss_3': laplacian3,
                        'normal_loss_1': normal1,
                        'normal_loss_2': normal2,
                        'normal_loss_3': normal3
                    }

                    if config['neptune']['activate']:
                        run["train/chamfer_distance_1"].log(chamfer1)
                        run["train/chamfer_distance_2"].log(chamfer2)
                        run["train/chamfer_distance_3"].log(chamfer3)
                        run["train/edge_loss_1"].log(edge1)
                        run["train/edge_loss_2"].log(edge2)
                        run["train/edge_loss_2"].log(edge3)
                        run["train/laplacia_loss_1"].log(laplacian1)
                        run["train/laplacia_loss_2"].log(laplacian2)
                        run["train/laplacia_loss_3"].log(laplacian3)
                        run["train/normal_loss_1"].log(normal1)
                        run["train/normal_loss_2"].log(normal2)
                        run["train/normal_loss_3"].log(normal3)
                    elif config['w&b']['activate']:
                        wandb.log(
                            loss
                        )

                mesh1_loss_on_batch = mesh1_loss_on_batch / conv64[i].shape()[0]
                mesh2_loss_on_batch = mesh2_loss_on_batch / conv128[i].shape()[0]
                mesh3_loss_on_batch = mesh3_loss_on_batch / conv256[i].shape()[0]
            else:
                raise NotImplementedError("At the moment of the code creation the batch is not supported by pytorch3d v 0.5.0.\n" +
                                          "Please contact the developer at mameli.1.marco@gmail.com with object PyTorch3D Pixel2Mesh Re-Implementation" +
                                          "and asck for code updates")
            # backpropagation # TODO understand the backpropagation see https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
            mesh1_loss_on_batch.backward()
            mesh2_loss_on_batch.backward()
            mesh3_loss_on_batch.backward()

            image_optimizer.step()
            mesh_optimizare.step()

        validation(config, convolutional_model=convolutional_model, graph_model=graph_model, validation_dataloader=validation_dataloader, device=device, run=run)