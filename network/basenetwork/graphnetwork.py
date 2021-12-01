"""The graph convolutional network for mesh deformation

:author: Marco Mameli
:return: Graph Convolutional neural network
:rtype: torch.nn.Module
"""
from torch import nn
from torch.nn import ModuleDict, ModuleList
from pytorch3d.ops import vert_align, GraphConv
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int , output_dim: int = 3, hidden_layer_count: int = 12, start_mesh = None) -> None:
        super(GraphConvolution, self).__init__()
        
        # Create the First GraphConvolutionBlock
        self.gc1_1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.gc_module_1 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc1_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        # GraphPooling
        self.gp1 = SubdivideMeshes(meshes=None)
        
        # Create the Second GraphConvolutionBlock
        self.gc2_1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.gc_module_2 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc2_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        # GraphPooling
        self.gp2 = SubdivideMeshes(meshes=None)
        
        # Create the Third GraphConvolutionBlock
        self.gc3_1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.gc_module_3 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc3_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        
    def forward(slef, conv_64, conv_128, conv_256, conv_512):
        