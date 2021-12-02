"""The graph convolutional network for mesh deformation

:author: Marco Mameli
:return: Graph Convolutional neural network
:rtype: torch.nn.Module
"""

from torch import nn
from torch._C import Graph
from torch.nn import ModuleList
from pytorch3d.ops import vert_align, GraphConv
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes

class GraphConvolution(nn.Module):
    """Graph Convolutional Network configuration

    :param nn: Inherit from nn.Module
    :type nn: nn.Module
    """
    def __init__(self, input_dim: int, hidden_dim: int , output_dim: int = 3, hidden_layer_count: int = 12, start_mesh: Meshes = None) -> None:
        super(GraphConvolution, self).__init__()
        
        self.vertices = start_mesh.verts_padded() #TODO insert the mesh vertices here (now None for the creation test.)
        self.mesh = start_mesh
        
        self.gc1_1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim) # First GraphConvolutional Block
        
        self.gc_module_1 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc1_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        self.gp1 = SubdivideMeshes(meshes=None) # Substitute the GraphPooling layer in the pixel2mesh TF implementation [the mesh is none because it is used the output from the gc1_14]
        
        self.gc2_1 = GraphConv(input_dim=input_dim + hidden_dim, output_dim=hidden_dim) # Second GraphConvolutional Block
        self.gc_module_2 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc2_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        self.gp2 = SubdivideMeshes(meshes=None) # Substitute the GraphPooling layer in the pixel2mesh TF implementation [the mesh is none because it is used the output from the gc2_14]
        
        self.gc3_1 = GraphConv(input_dim=input_dim + hidden_dim, output_dim=hidden_dim) # Third GraphConvolutional Block
        self.gc_module_3 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc3_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        
    def forward(self, conv64, conv128, conv256, conv512 = None, device="cuda"):
        """The forward function

        :param conv64: the features tensor from the conv layer with 64 dim output
        :type conv64: torch.Tensor
        :param conv128: the features tensor from the conv layer with 128 dim output
        :type conv128: torch.Tensor
        :param conv256: the features tensor from the conv layer with 256 dim output
        :type conv256: torch.Tensor
        :param conv512: the features tensor from the conv layer with 512 dim output (not used)
        :type conv512: torch.Tensor
        :return: the output of the convolution
        :rtype: torch.Tensor (it can be changed in pytorch3d.io.Mesh)
        """
        vert_features_from_conv64 = vert_align(conv64, self.mesh)
        
        x = self.gc1_1(vert_features_from_conv64[0], self.mesh.edges_packed())
        
        for i, l in enumerate(self.gc_module_1):
            x = self.gc_module_1[i](x, self.mesh.edges_packed())
        
        x = self.gc1_14(x, self.mesh.edges_packed())
        
        #TODO check reconstruct the mesh with the new vetices
        
        self.mesh = Meshes(verts=[x], faces=self.mesh.faces_list())
        
        x = self.gp1(self.mesh)
        
        vert_features_from_conv64 = vert_align(conv128, self.mesh) #TODO change the self.mesh to the output of the pooling.
        
        x = self.gc2_1(vert_features_from_conv64) #TODO here the features are to be concatenated
        
        for i, l in enumerate(self.gc_module_2):
            x = self.gc_module_2[i](x)
            
        x = self.gc2_14(x)
        
        x = self.gp2(x)
        
        vert_features_from_conv64 = vert_align(conv256, self.mesh) #TODO change the self.mesh to the output of the pooling.
        
        x = self.gc3_1(vert_features_from_conv64) #TODO here the features are to be concatenated
        
        for i, l in enumerate(self.gc_module_3):
            x = self.gc_module_3[i](x)
        
        x = self.gc3_14(x)
        
        return x
    
if __name__ == "__main__":
    """Network Testing
    """
    import torch
    from pytorch3d.io import load_obj, load_objs_as_meshes
    from pytorch3d.structures.meshes import Meshes
    from pytorch3d.utils import ico_sphere
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    
    ico = ico_sphere(3).to(device=device)
    #print("Verts list", ico.verts_list())
    #print("Faces list", ico.faces_list())
    #print("Verts packed", ico.verts_packed())
    
    conv64 = torch.rand([1, 64, 54, 54], dtype=torch.float32, device=device)
    conv128 = torch.rand([1, 128, 26, 26], dtype=torch.float32, device=device)
    conv256 = torch.rand([1, 256, 12, 12], dtype=torch.float32, device=device)
    conv512 = torch.rand([1, 512, 3, 3], dtype=torch.float32, device=device)
    gcn = GraphConvolution(input_dim=64, hidden_dim=256 , output_dim = 3, start_mesh=ico) # Need to understand the input dimension and the internal dimension how work
    gcn = gcn.to(device=device) # Move on GPU
    #print(GraphConvolution)
    output = gcn(conv64, conv128, conv256)
    