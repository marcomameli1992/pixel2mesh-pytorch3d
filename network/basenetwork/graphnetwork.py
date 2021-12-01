"""The graph convolutional network for mesh deformation

:author: Marco Mameli
:return: Graph Convolutional neural network
:rtype: torch.nn.Module
"""
from torch import nn
from pytorch3d.ops import vert_align, GraphConv

class GraphConvolution(nn.Module):
    def __init__(self) -> None:
        super(GraphConvolution, self).__init__()
