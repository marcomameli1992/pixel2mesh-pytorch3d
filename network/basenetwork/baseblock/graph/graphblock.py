"""The graph convolutional block basic structure

:author: Marco Mameli
:return: Graph Convolutional neural network
:rtype: torch.nn.Module
"""

from torch import nn
from torch.nn import ReLU
from torch.nn import ModuleList
from pytorch3d.ops import GraphConv

class GraphConvolutionalBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layer_count: int = 12):
        super(GraphConvolutionalBlock, self).__init__()
        
        self.activation = ReLU()
        
        self.gc1 = GraphConv(input_dim=input_dim, output_dim=output_dim)
        
        self.gc_module = ModuleList(
            [GraphConv(input_dim=output_dim, output_dim=output_dim) for n in range(hidden_layer_count)]
        )
        self.gc14 = GraphConv(input_dim=output_dim, output_dim=output_dim)
    
    def forward(self, features, edges):
        
        residual1_1 = x = self.activation(self.gc1_1(features, edges))
        
        for i, l in enumerate(self.gc_module):
            x = self.gc_module_1[i](x, edges)
            x = self.activation(x)
        
        x = residual1_1 + x
        
        vertices = self.activation(self.gc1_14(x, edges))
        
        return vertices