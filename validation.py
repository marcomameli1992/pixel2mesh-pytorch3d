"""

"""

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from pytorch3d import loss as l3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
import neptune.new as neptune

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

            mesh1_loss_on_batch = 0.0
            mesh2_loss_on_batch = 0.0
            mesh3_loss_on_batch = 0.0
            if len(conv16.shape()) == 4 or len(conv32.shape()) == 4 or len(conv64.shape()) == 4 or \
                    len(conv128.shape()) == 4 or len(conv256.shape()) == 4 or len(conv512.shape()) == 4:
                # Batch size different from 1
                for i in conv64.shape()[0]:
                    # opening mesh for the image in the batch
                    label_mesh_path = data['img_path'].replace(config['img_base_path'],
                                                               config['obj_base_path']).replace(
                        '/rendering/' + config['img_name'] + '.png', '/models/' + config['obj_name'] + '.obj')

                    label_mesh1 = load_objs_as_meshes([label_mesh_path], device=device)
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
                    elif config['w&b']['activate']:
                        wandb.log(
                            evaluate
                        )