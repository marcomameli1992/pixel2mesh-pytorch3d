"""The graph convolutional network for mesh deformation

:author: Marco Mameli
:return: Graph Convolutional neural network
:rtype: torch.nn.Module
"""
from torch import nn
from torch.nn import ModuleList
from pytorch3d.ops import vert_align, GraphConv
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int , output_dim: int = 3, hidden_layer_count: int = 12, start_mesh = None) -> None:
        super(GraphConvolution, self).__init__()
        
        #TODO Extracting the vertex from the starting mesh 
        self.vertices = None 
        
        # Create the First GraphConvolutionBlock
        self.gc1_1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.gc_module_1 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc1_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        # GraphPooling
        self.gp1 = SubdivideMeshes(meshes=None)
        
        # Create the Second GraphConvolutionBlock
        self.gc2_1 = GraphConv(input_dim=input_dim + hidden_dim, output_dim=hidden_dim)
        self.gc_module_2 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc2_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        # GraphPooling
        self.gp2 = SubdivideMeshes(meshes=None)
        
        # Create the Third GraphConvolutionBlock
        self.gc3_1 = GraphConv(input_dim=input_dim + hidden_dim, output_dim=hidden_dim)
        self.gc_module_3 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc3_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        
    def forward(self, conv64, conv128, conv256, conv512):
        vert_features_from_conv64 = vert_align(conv64, self.vertices)
        
        x = self.gc1_1(vert_features_from_conv64)
        
        for i, l in enumerate(self.gc_module_1):
            x = self.gc_module_1[i](x)
        
        x = self.gc1_14(x)
        
        vert_features_from_conv64 = vert_align(conv128, self.vertices)
        
        x = self.gc2_1(vert_features_from_conv64) #TODO here the features are to be concatenated
        
        for i, l in enumerate(self.gc_module_2):
            x = self.gc_module_2[i](x)
            
        x = self.gc2_14(x)
        
        vert_features_from_conv64 = vert_align(conv256, self.vertices)
        
        x = self.gc3_1(vert_features_from_conv64)
        
        for i, l in enumerate(self.gc_module_3):
            x = self.gc_module_3[i](x)
        
        x = self.gc3_14(x)
        
        return x