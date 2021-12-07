"""

"""

import torch
import wandb
from pytorch3d.structures import Meshes
from torch import nn
from torch.utils.data import DataLoader
from pytorch3d import loss as l3d
from pytorch3d.io import save_obj, load_obj
from pytorch3d.utils import ico_sphere
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
import neptune.new as neptune
from utils.plotting import plot_pointcloud

def validation(config, convolutional_model: nn.Module, graph_model: nn.Module, validation_dataloader: DataLoader, device:str, run=None) -> None:
    """
    The validation function for validation step during the training
    Args:
        config: the configuration structure from the JSON
        convolutional_model: the convolutional model
        graph_model: the graph convolutional model
        validation_dataloader: the validation data loader
        run: the neptune client

    Returns:

    """

    # Create the subdivision for label mesh
    subdivide = SubdivideMeshes()

    # define the optimizer
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):

            # forward + backward + optimize
            conv16, conv32, conv64, conv128, conv256, conv512 = convolutional_model(data['image'])



            if len(conv16.shape()) == 4 or len(conv32.shape()) == 4 or len(conv64.shape()) == 4 or \
                    len(conv128.shape()) == 4 or len(conv256.shape()) == 4 or len(conv512.shape()) == 4:
                # Batch size different from 1
                for i in conv64.shape()[0]:
                    # opening mesh for the image in the batch
                    label_mesh_path = data['img_path'].replace(config['img_base_path'],
                                                               config['obj_base_path']).replace(
                        '/rendering/' + config['img_name'] + '.png', '/models/' + config['obj_name'] + '.obj')

                    # Read the target 3D model using load_obj
                    verts, faces, aux = load_obj(label_mesh_path)

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
                    label_mesh1 = Meshes(verts=[verts], faces=[faces_idx])

                    # Subdivide label mesh for the losses
                    label_mesh2 = subdivide(label_mesh1)
                    label_mesh3 = subdivide(label_mesh2)

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
                    label_mesh2 = subdivide(label_mesh1)
                    label_mesh3 = subdivide(label_mesh2)

                    mesh = ico_sphere(level=0, device=device)

                    mesh1, mesh2, mesh3 = graph_model(mesh, conv64[i].unsqueeze(0), conv128[i].unsqueeze(0),
                                                      conv256[i].unsqueeze(0))

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

                    # Evaluate
                    point_medhe_edge_distance_1 = l3d.point_mesh_edge_distance([mesh1],[point_mesh1])
                    point_medhe_edge_distance_2 = l3d.point_mesh_edge_distance([mesh2], [point_mesh2])
                    point_medhe_edge_distance_3 = l3d.point_mesh_edge_distance([mesh3], [point_mesh3])

                    point_medhe_face_distance_1 = l3d.point_mesh_face_distance([mesh1], [point_mesh1])
                    point_medhe_face_distance_2 = l3d.point_mesh_face_distance([mesh2], [point_mesh2])
                    point_medhe_face_distance_3 = l3d.point_mesh_face_distance([mesh3], [point_mesh3])

                    evaluate = {
                        'val_chamfer_distance_1': chamfer1,
                        'val_chamfer_distance_2': chamfer2,
                        'val_chamfer_distance_3': chamfer3,
                        'val_point_medhe_edge_distance_1': point_medhe_edge_distance_1,
                        'val_point_medhe_edge_distance_2': point_medhe_edge_distance_2,
                        'val_point_medhe_edge_distance_3': point_medhe_edge_distance_3,
                        'val_point_medhe_face_distance_1': point_medhe_face_distance_1,
                        'val_point_medhe_face_distance_2': point_medhe_face_distance_2,
                        'val_point_medhe_face_distance_3': point_medhe_face_distance_3,
                    }

                    if run != None and config['neptune']['activate']:
                        run["validation/chamfer_distance_1"].log(chamfer1)
                        run["validation/chamfer_distance_2"].log(chamfer2)
                        run["validation/chamfer_distance_3"].log(chamfer3)
                        run["validation/point_medhe_edge_distance_1"].log(point_medhe_edge_distance_1)
                        run["validation/point_medhe_edge_distance_2"].log(point_medhe_edge_distance_2)
                        run["validation/point_medhe_edge_distance_3"].log(point_medhe_edge_distance_3)
                        run["validation/point_medhe_face_distance_1"].log(point_medhe_face_distance_1)
                        run["validation/point_medhe_face_distance_2"].log(point_medhe_face_distance_2)
                        run["validation/point_medhe_face_distance_3"].log(point_medhe_face_distance_3)
                        run["validation/label_mesh1"].log(plot_pointcloud(label_mesh1))
                        run["validation/label_mesh2"].log(plot_pointcloud(label_mesh2))
                        run["validation/label_mesh3"].log(plot_pointcloud(label_mesh3))
                        run["validation/generated_mesh1"].log(plot_pointcloud(mesh1))
                        run["validation/generated_mesh2"].log(plot_pointcloud(mesh2))
                        run["validation/generated_mesh3"].log(plot_pointcloud(mesh3))
                    elif config['w&b']['activate']:
                        wandb.log(
                            evaluate
                        )
                        # saving mesh to file for the logging
                        save_obj(f=config['save_obj'] + 'generated_mesh1.obj', verts=mesh1.verts_list(), faces=mesh1.faces_list())
                        save_obj(f=config['save_obj'] + 'generated_mesh2.obj', verts=mesh2.verts_list(),
                                 faces=mesh2.faces_list())
                        save_obj(f=config['save_obj'] + 'generated_mesh3.obj', verts=mesh3.verts_list(),
                                 faces=mesh3.faces_list())
                        save_obj(f=config['save_obj'] + 'subdivided_label_mesh1.obj', verts=label_mesh2.verts_list(),
                                 faces=label_mesh2.faces_list())
                        save_obj(f=config['save_obj'] + 'subdivided_label_mesh2.obj', verts=label_mesh3.verts_list(),
                                 faces=label_mesh3.faces_list())
                        wandb.log({
                            "validation_label_mesh1": wandb.Object3D(open(label_mesh_path)),
                            "validation_label_mesh2": wandb.Object3D(open(config['save_obj'] + 'subdivided_label_mesh1.obj')),
                            "validation_label_mesh3": wandb.Object3D(open(config['save_obj'] + 'subdivided_label_mesh2.obj')),
                            "validation_generated_mesh1": wandb.Object3D(open(config['save_obj'] + 'generated_mesh1.obj')),
                            "validation_generated_mesh2": wandb.Object3D(
                                open(config['save_obj'] + 'generated_mesh2.obj')),
                            "validation_generated_mesh3": wandb.Object3D(
                                open(config['save_obj'] + 'generated_mesh3.obj')),
                        })