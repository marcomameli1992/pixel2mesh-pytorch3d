"""Mesh Deformation Block
:author: Marco Mameli
:return: Mesh deformation block as layer for a neural network
:rtype: torch.nn.Module
"""
from torch import nn
from pytorch3d.ops import vert_align, GraphConv

def vertex_feature_from_image_feature(img_feature, vertex_features):
    """This function uses the vert_align function to convert from image features 
    to vertex features

    :param img_feature: feature map from image
    :type img_feature: torch.Tensor
    :param vertex_features: [description]
    :type vertex_features: [type]
    :return: [description]
    :rtype: [type]
    """
    pixel2vert_features = vert_align(img_feature, verts=vertex_features)
    
    return pixel2vert_features 