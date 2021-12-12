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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, hidden_layer_count: int = 12):
        super(GraphConvolutionalBlock, self).__init__()
        
        self.activation = ReLU()
        
        self.gc1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        
        self.gc_module = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
    
    def forward(self, features, edges):
        
        residual = x = self.activation(self.gc1(features, edges))
        
        for i, l in enumerate(self.gc_module):
            x = self.activation(self.gc_module[i](x, edges))
        
        auxiliary = x
        
        x = residual + x
        
        vertices = self.gc14(x, edges)
        
        return vertices, auxiliary