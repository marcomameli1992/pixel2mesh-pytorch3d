"""The graph convolutional network for mesh deformation

:author: Marco Mameli
:return: Graph Convolutional neural network
:rtype: torch.nn.Module
"""

from pytorch3d.structures import meshes
from torch.nn import Module
from torch.nn import ReLU
from pytorch3d.ops import vert_align
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes
from network.basenetwork.baseblock.graph.graphblock import GraphConvolutionalBlock

class GraphConvolutional(Module):
    """Graph Convolutional Network configuration

    :param Module: torch module for network
    :type Module: Module type
    """
    def __init__(self, first_block_input_dim: int, second_block_input_dim: int, third_block_input_dim, hidden_dim: int, output_dim: int, hidden_layer_count: int = 12):
        super(GraphConvolutional, self).__init__()
        
        # Activation function
        self.activation = ReLU()
        
        # First Graph Block
        self.gb1 = GraphConvolutionalBlock(input_dim=first_block_input_dim, hidden_dim=hidden_dim, hidden_layer_count=hidden_layer_count, output_dim=output_dim)
        
        # Mesh expansion
        self.gp1 = SubdivideMeshes()
        
        # Second Graph Block
        self.gb2 = GraphConvolutionalBlock(input_dim=second_block_input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hidden_layer_count=hidden_layer_count)
        
        # Mesh expansion
        self.gp2 = SubdivideMeshes()
        
        # Third Graph Block
        self.gb3 = GraphConvolutionalBlock(input_dim=third_block_input_dim, hidden_layer_count=hidden_layer_count, hidden_dim=hidden_dim, output_dim=output_dim)
        
    def forward(self, mesh: Meshes, conv_features_64, conv_features_128, conv_features_256, conv_features_512=None):
        """Forward function for the use of the model

        :param mesh: [description]
        :type mesh: [type]
        :param conv_features_64: [description]
        :type conv_features_64: [type]
        :param conv_features_128: [description]
        :type conv_features_128: [type]
        :param conv_features_256: [description]
        :type conv_features_256: [type]
        :param conv_features_512: [description], defaults to None
        :type conv_features_512: [type], optional
        """
        
        # Vertex projection on features
        vertex_feature_from_conv_64 = vert_align(conv_features_64, mesh)
        
        # Apply Graph Convolution
        vertices1, features1 = self.gb1(vertex_feature_from_conv_64[0], mesh.edges_packed())
        
        # Get The new Meshes
        mesh1 = mesh = Meshes(verts=[vertices1], faces=mesh.faces_list())

        # Free memory for unusefull data
        del vertices1
        del features1
        
        # Mesh density increase
        mesh = self.gp1(mesh)

        # Vertex projection on features
        vertex_feature_from_conv_128 = vert_align(conv_features_128, mesh)

        # Apply Graph Convolution
        vertices2, features2 = self.gb2(vertex_feature_from_conv_128[0], mesh.edges_packed())
        
        # Get The new Meshes
        mesh2 = mesh = Meshes(verts=[vertices2], faces=mesh.faces_list())

        # Free memory for unusefull data
        del vertices2
        del features2
        
        # Mesh density increase
        mesh = self.gp2(mesh)
        
        # Vertex projection on features
        vertex_feature_from_conv_256 = vert_align(conv_features_256, mesh)
        
        # Apply Graph Convolution
        vertices3, features3 = self.gb3(vertex_feature_from_conv_256[0], mesh.edges_packed())

        # Get The new Meshes
        mesh3 = Meshes(verts=[vertices3], faces=mesh.faces_list())

        # Free memory for unusefull data
        del vertices3
        del features3
        
        return mesh1, mesh2, mesh3
    

if __name__ == "__main__":
    """Network testing
    """
    import torch
    from pytorch3d.structures.meshes import Meshes
    from pytorch3d.utils import ico_sphere
    from pytorch3d.structures import join_meshes_as_batch 
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    ico = ico_sphere(3).to(device=device)
    
    mesh_batch = join_meshes_as_batch([ico, ico])
    
    conv64 = torch.rand((2, 64, 54, 54), dtype=torch.float32, device=device)
    conv128 = torch.rand((2, 128, 26, 26), dtype=torch.float32, device=device)
    conv256 = torch.rand((2, 256, 12, 12), dtype=torch.float32, device=device)
    conv512 = torch.rand((2, 512, 3, 3), dtype=torch.float32, device=device)
    
    gcn = GraphConvolutional(first_block_input_dim=64, second_block_input_dim=128, third_block_input_dim=256, hidden_dim=256 , output_dim = 3) # Need to understand the input dimension and the internal dimension how work
    gcn = gcn.to(device=device) # Move on GPU
    
    for i in range(2):
        mesh1, mesh2, mesh3 = gcn(ico, conv64[0].unsqueeze(0), conv128[0].unsqueeze(0), conv256[0].unsqueeze(0))