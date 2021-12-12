"""
Training function for the network configuration
:author: Marco Mameli

"""
import torch
import wandb
from pytorch3d.structures import Meshes
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from pytorch3d import loss as l3d
from torch import cuda
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.utils import ico_sphere
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
import neptune.new as neptune
import numpy as np
from validation import validation
import tqdm

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

    # model on GPU
    device = torch.device("cuda:0") if cuda.is_available() else torch.device("cpu")
    convolutional_model.to(device)
    graph_model.to(device)

    #% Initialize experiment history.
    # Check for neptune
    run = None
    if config['neptune']['activate']:
        neptune.create_experiment(send_hardware_metrics=config['neptune']['hw_metrics_active'])
        run = neptune.init(
            project=config['neptune']['project'],
            api_token=config['neptune']['api_token']
        )
    elif config['w&b']['activate']:
        run = wandb.init(project=config['w&b']['project'], entity=config['w&b']['entity'], reinit=True)

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
        wandb.config = params
    # optmizer init
    if config['training']['optimizer'] == "sgd":
        image_optimizer = optim.SGD(convolutional_model.parameters(), lr=config['training']['learning_rate'], momentum=config['training']['momentum'])
        mesh_optimizare = optim.SGD(graph_model.parameters(), lr=config['training']['learning_rate'], momentum=config['training']['momentum'])
    elif config['training']['optimizer'] == "adam":
        image_optimizer = optim.Adam(convolutional_model.parameters(), lr=config['training']['learning_rate'])
        mesh_optimizare = optim.Adam(graph_model.parameters(), lr=config['training']['learning_rate'])
    else:
        raise NotImplementedError("Please in the config.json you have to specify an optimizer in [sgd, adam]")

    # Create the subdivision for label mesh
    subdivide = SubdivideMeshes()

    # define the optimizer
    for epoch in tqdm.tqdm(range(config['training']['epochs'])):
        validation_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(train_dataloader)):
            # reset grad
            image_optimizer.zero_grad()
            mesh_optimizare.zero_grad()
            
            # forward + backward + optimize
            conv16, conv32, conv64, conv128, conv256, conv512 = convolutional_model(data['image'].type(torch.float32).to(device))

            mesh1_loss_on_batch = 0.0
            mesh2_loss_on_batch = 0.0
            mesh3_loss_on_batch = 0.0

            penalty1 = 1.0
            penalty2 = 1.0
            penalty3 = 1.0

            if len(conv16.shape) == 4 or len(conv32.shape) == 4 or len(conv64.shape) == 4 or \
                    len(conv128.shape) == 4 or len(conv256.shape) == 4 or len(conv512.shape) == 4:
                # Batch size different from 1
                for i in range(conv64.shape[0]):

                    # opening mesh for the image in the batch
                    label_mesh_path = data['img_path'][i].replace(config['img_base_path'], config['obj_base_path']).replace(
                        '/rendering/' + config['img_name'] + '.png', '/models/' + config['obj_name'] + '.obj')

                    if config['debug']:
                        print(f"Epoch {epoch} - Batch index: {i}")
                        print(label_mesh_path)

                    # Read the target 3D model using load_obj
                    verts, faces, aux = load_obj(label_mesh_path)

                    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
                    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
                    # For this tutorial, normals and textures are ignored.
                    faces_idx = faces.verts_idx.to(device)
                    verts = verts.to(device)

                    #print("label_mesh1 verts tensor: ", verts.shape)

                    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
                    # (scale, center) will be used to bring the predicted mesh to its original center and scale
                    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
                    # as suggested in the pytorch3D tutorials: https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
                    center = verts.mean(0)
                    verts = verts - center
                    scale = max(verts.abs().max(0)[0])
                    verts = verts / scale

                    # reconstruct the mesh
                    label_mesh1 = Meshes(verts=[verts], faces=[faces_idx])

                    # Subdivide label mesh for the losses
                    label_mesh2 = subdivide(label_mesh1)
                    label_mesh3 = subdivide(label_mesh2)

                    if config['debug']:

                        print("label_mesh2 verts tensor: ", label_mesh2.verts_list()[0].shape)
                        print("label_mesh3 verts tensor: ", label_mesh3.verts_list()[0].shape)

                        print("label_mesh1 nan verts tensor: ", torch.any(label_mesh1.verts_list()[0].isnan()))
                        print("label_mesh2 nan verts tensor: ", torch.any(label_mesh2.verts_list()[0].isnan()))
                        print("label_mesh3 nan verts tensor: ", torch.any(label_mesh3.verts_list()[0].isnan()))

                        print("label_mesh1 area: ", label_mesh1.faces_areas_packed().max())
                        print("label_mesh2 area: ", label_mesh2.faces_areas_packed().max())
                        print("label_mesh3 area: ", label_mesh3.faces_areas_packed().max())


                        #print("label_mesh1 nan verts tensor: ", label_mesh1.verts_list()[0].isnan())
                        #print("label_mesh2 nan verts tensor: ", label_mesh2.verts_list()[0].isnan())
                        #print("label_mesh3 nan verts tensor: ", label_mesh3.verts_list()[0].isnan())

                    if config['starting_mesh']['path'] != "None":
                        # Read the target 3D model using load_obj
                        verts, faces, aux = load_obj(config['starting_mesh']['path'])

                        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
                        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
                        # For this tutorial, normals and textures are ignored.
                        faces_idx = faces.verts_idx.to(device)
                        verts = verts.to(device)

                        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
                        # (scale, center) will be used to bring the predicted mesh to its original center and scale
                        # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
                        # as suggested in the pytorch3D tutorials: https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
                        center = verts.mean(0)
                        verts = verts - center
                        scale = max(verts.abs().max(0)[0])
                        verts = verts / scale

                        # reconstruct the mesh
                        mesh = Meshes(verts=[verts], faces=[faces_idx])
                    else:
                        mesh = ico_sphere(level=config['starting_mesh']['ico_sphere_subdivide_level'], device=device)
                        if config['debug']:
                            print("ico sphere area: ", mesh.faces_areas_packed().max())


                    mesh1, mesh2, mesh3 = graph_model(mesh, conv64[i].unsqueeze(0), conv128[i].unsqueeze(0), conv256[i].unsqueeze(0))

                    # compute area of the meshes

                    if config['debug']:

                        print("generated_mesh1 nan verts tensor: ", torch.any(mesh1.verts_list()[0].isnan()))
                        print("generated_mesh2 nan verts tensor: ", torch.any(mesh2.verts_list()[0].isnan()))
                        print("generated_mesh3 nan verts tensor: ", torch.any(mesh3.verts_list()[0].isnan()))

                        #print("generated_mesh1 nan verts tensor: ", mesh1.verts_list()[0].isnan())
                        #print("generated_mesh2 nan verts tensor: ", mesh2.verts_list()[0].isnan())
                        #print("generated_mesh3 nan verts tensor: ", mesh3.verts_list()[0].isnan())

                        print("generated_mesh1 verts tensor: ", mesh1.verts_list()[0].shape)
                        print("generated_mesh2 verts tensor: ", mesh2.verts_list()[0].shape)
                        print("generated_mesh3 verts tensor: ", mesh3.verts_list()[0].shape)

                        print("generated_mesh1 area: ", mesh1.faces_areas_packed().max())
                        print("generated_mesh2 area: ", mesh2.faces_areas_packed().max())
                        print("generated_mesh3 area: ", mesh3.faces_areas_packed().max())

                        torch.save(mesh1.verts_list(), config["save_obj"] + f'mesh1_{i}.pt')
                        torch.save(mesh2.verts_list(), config["save_obj"] + f'mesh2_{i}.pt')
                        torch.save(mesh3.verts_list(), config["save_obj"] + f'mesh3_{i}.pt')

                    if mesh1.faces_areas_packed().max() == 0 and mesh2.faces_areas_packed().max() != 0 and mesh3.faces_areas_packed().max() != 0:
                        mesh1 = mesh2
                        penalty1 = 100.0
                    if mesh1.faces_areas_packed().max() != 0 and mesh2.faces_areas_packed().max() == 0 and mesh3.faces_areas_packed().max() != 0:
                        mesh2 = mesh3
                        penalty2 = 100.0
                    if mesh1.faces_areas_packed().max() != 0 and mesh2.faces_areas_packed().max() != 0 and mesh3.faces_areas_packed().max() == 0:
                        mesh3 = mesh2
                        penalty3 = 100.0
                    if mesh1.faces_areas_packed().max() == 0 and mesh2.faces_areas_packed().max() == 0 and mesh3.faces_areas_packed().max() != 0:
                        mesh1 = mesh3
                        mesh2 = mesh3
                        penalty1 = 100.0
                        penalty2 = 100.0
                    if mesh1.faces_areas_packed().max() != 0 and mesh2.faces_areas_packed().max() == 0 and mesh3.faces_areas_packed().max() == 0:
                        mesh3 = mesh1
                        mesh2 = mesh1
                        penalty3 = 100.0
                        penalty2 = 100.0
                    if mesh1.faces_areas_packed().max() == 0 and mesh2.faces_areas_packed().max() != 0 and mesh3.faces_areas_packed().max() == 0:
                        mesh1 = mesh2
                        mesh3 = mesh2
                        penalty1 = 100.0
                        penalty2 = 100.0
                    # point cloud conversion
                    point_label_mesh1 = sample_points_from_meshes(label_mesh1, num_samples=config['point_sampling_value'])
                    point_label_mesh2 = sample_points_from_meshes(label_mesh2, num_samples=config['point_sampling_value'])
                    point_label_mesh3 = sample_points_from_meshes(label_mesh3, num_samples=config['point_sampling_value'])

                    point_mesh1 = sample_points_from_meshes(mesh1, num_samples=config['point_sampling_value'])
                    point_mesh2 = sample_points_from_meshes(mesh2, num_samples=config['point_sampling_value'])
                    point_mesh3 = sample_points_from_meshes(mesh3, num_samples=config['point_sampling_value'])

                    # chamfer distance
                    chamfer1, _ = l3d.chamfer_distance(point_mesh1, point_label_mesh1)
                    chamfer2, _ = l3d.chamfer_distance(point_mesh2, point_label_mesh2)
                    chamfer3, _ = l3d.chamfer_distance(point_mesh3, point_label_mesh3)

                    mesh1_loss_on_batch += chamfer1 * config['loss_weights']['chamfer'] * penalty1
                    mesh2_loss_on_batch += chamfer2 * config['loss_weights']['chamfer'] * penalty2
                    mesh3_loss_on_batch += chamfer3 * config['loss_weights']['chamfer'] * penalty3

                    # edge smooth
                    edge1 = l3d.mesh_edge_loss(mesh1)
                    edge2 = l3d.mesh_edge_loss(mesh2)
                    edge3 = l3d.mesh_edge_loss(mesh3)
                    mesh1_loss_on_batch += edge1 * config['loss_weights']['edge'] * penalty1
                    mesh2_loss_on_batch += edge2 * config['loss_weights']['edge'] * penalty2
                    mesh3_loss_on_batch += edge3 * config['loss_weights']['edge'] * penalty3

                    # laplacian smooth
                    laplacian1 = l3d.mesh_laplacian_smoothing(mesh1)
                    laplacian2 = l3d.mesh_laplacian_smoothing(mesh2)
                    laplacian3 = l3d.mesh_laplacian_smoothing(mesh3)
                    mesh1_loss_on_batch += laplacian1 * config['loss_weights']['laplacian'] * penalty1
                    mesh2_loss_on_batch += laplacian2 * config['loss_weights']['laplacian'] * penalty2
                    mesh3_loss_on_batch += laplacian3 * config['loss_weights']['laplacian'] * penalty3

                    # normal convalidation
                    normal1 = l3d.mesh_normal_consistency(mesh1)
                    normal2 = l3d.mesh_normal_consistency(mesh2)
                    normal3 = l3d.mesh_normal_consistency(mesh3)
                    mesh1_loss_on_batch += normal1 * config['loss_weights']['normal'] * penalty1
                    mesh2_loss_on_batch += normal2 * config['loss_weights']['normal'] * penalty2
                    mesh3_loss_on_batch += normal3 * config['loss_weights']['normal'] * penalty3

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
                            {
                                'train_chamfer_distance_1': chamfer1,
                                'train_chamfer_distance_2': chamfer2,
                                'train_chamfer_distance_3': chamfer3,
                                'train_edge_loss_1': edge1,
                                'train_edge_loss_2': edge2,
                                'train_edge_loss_3': edge3,
                                'train_laplacia_loss_1': laplacian1,
                                'train_laplacia_loss_2': laplacian2,
                                'train_laplacia_loss_3': laplacian3,
                                'train_normal_loss_1': normal1,
                                'train_normal_loss_2': normal2,
                                'train_normal_loss_3': normal3
                            }
                        )

                # TODO insert the graphs for the reconstruction

                mesh1_loss_on_batch = mesh1_loss_on_batch / conv64[i].shape[0]
                mesh2_loss_on_batch = mesh2_loss_on_batch / conv128[i].shape[0]
                mesh3_loss_on_batch = mesh3_loss_on_batch / conv256[i].shape[0]
            else:
                raise NotImplementedError("At the moment of the code creation the batch is not supported by pytorch3d v 0.5.0.\n" +
                                          "Please contact the developer at mameli.1.marco@gmail.com with object PyTorch3D Pixel2Mesh Re-Implementation" +
                                          "and asck for code updates")
            # backpropagation # TODO understand the backpropagation see https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
            mesh_loss_on_batch = mesh1_loss_on_batch + mesh2_loss_on_batch + mesh3_loss_on_batch
            #mesh1_loss_on_batch.backward()
            #mesh2_loss_on_batch.backward()
            #mesh3_loss_on_batch.backward()

            mesh_loss_on_batch.backward()

            image_optimizer.step()
            mesh_optimizare.step()

        new_validation_loss = validation(config, convolutional_model=convolutional_model, graph_model=graph_model, validation_dataloader=validation_dataloader, device=device, run=run)
        if new_validation_loss < validation_loss:
            # TODO save weight

            validation_loss = new_validation_loss
