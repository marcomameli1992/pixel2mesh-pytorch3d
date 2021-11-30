"""Mesh Deformation Block
:author: Marco Mameli
:return: Mesh deformation block as layer for a neural network
:rtype: torch.nn.Module
"""
from torch import nn
from pytorch3d.ops import vert_align, GraphConv

class MeshDeformationBlock(nn.Module):
    def __init__(self, graph_input_dim: int, graph_output_dim: int) -> None:
        """Init of the layer
        :param graph_input_dim: the input dimension for the graph convolution
        :type graph_input_dim: int
        :param graph_output_dim: the output dimension for the graph convolutional
        :type graph_output_dim: int
        """        
        super(MeshDeformationBlock, self).__init__()
        self.graph_conv1 = GraphConv(input_dim=graph_input_dim, output_dim=graph_output_dim)
        self.graph_conv2 = GraphConv(input_dim=graph_input_dim, output_dim=graph_output_dim)
        self.graph_conv3 = GraphConv(input_dim=graph_input_dim, output_dim=graph_output_dim)
    def forward(self, img_features, vertex_position, vertex_padded):
        """Forward pass for the Mesh Deformation Block

        :param img_features: the images features extracted from an image (using CNN)
        :type img_features: torch.tensor
        :param vertex_position: the position of the vertices in a mesh
        :type vertex_position: pytorch3d.vertex_padded
        :return: Tensor of features applied on a vertices of a mesh
        :rtype: torch.tensor
        """
        pix2vert_features = vert_align(img_features, verts=vertex_position)
        graph_input = pix2vert_features + vertex_padded
        mesh_deformation = self.graph_conv1(graph_input)
        mesh_deformation1 = self.graph_conv2(mesh_deformation)
        mesh_deformation1 = self.graph_conv3(mesh_deformation1)
        mesh_deformation_output = mesh_deformation + mesh_deformation1
        return mesh_deformation_output

 