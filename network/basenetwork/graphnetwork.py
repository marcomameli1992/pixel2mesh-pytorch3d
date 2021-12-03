"""The graph convolutional network for mesh deformation

:author: Marco Mameli
:return: Graph Convolutional neural network
:rtype: torch.nn.Module
"""

from torch import nn
from torch.nn import ReLU
from torch.nn import ModuleList
from pytorch3d.ops import vert_align, GraphConv
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes

class GraphConvolution(nn.Module):
    """Graph Convolutional Network configuration

    :param nn: Inherit from nn.Module
    :type nn: nn.Module
    """
    def __init__(self, input_dim: int, hidden_dim: int , output_dim: int = 3, hidden_layer_count: int = 12) -> None:
        super(GraphConvolution, self).__init__()
        
        self.activation = ReLU()
        
        self.gc1_1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim) # First GraphConvolutional Block
        
        self.gc_module_1 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc1_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        self.gp1 = SubdivideMeshes(meshes=None) # Substitute the GraphPooling layer in the pixel2mesh TF implementation [the mesh is none because it is used the output from the gc1_14]
        
        self.gc2_1 = GraphConv(input_dim=128, output_dim=hidden_dim) # Second GraphConvolutional Block
        self.gc_module_2 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc2_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        self.gp2 = SubdivideMeshes(meshes=None) # Substitute the GraphPooling layer in the pixel2mesh TF implementation [the mesh is none because it is used the output from the gc2_14]
        
        self.gc3_1 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) # Third GraphConvolutional Block
        self.gc_module_3 = ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for n in range(hidden_layer_count)]
        )
        self.gc3_14 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        
        
    def forward(self, mesh, conv64, conv128, conv256, conv512 = None):
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
        
        #TODO Check if it is possible to use batches for the moment not is possible
        
        vert_features_from_conv64 = vert_align(conv64, mesh)
        
        residual1_1 = x = self.activation(self.gc1_1(vert_features_from_conv64, mesh.edges_packed()))
        
        for i, l in enumerate(self.gc_module_1):
            x = self.gc_module_1[i](x, mesh.edges_packed())
            x = self.activation(x)
        
        x = residual1_1 + x
        
        vertices = self.activation(self.gc1_14(x, mesh.edges_packed()))
        
        #TODO check reconstruct the mesh with the new vetices
        
        mesh_gc_1 = self.mesh = Meshes(verts=[vertices], faces=mesh.faces_list())
        
        self.mesh, features_gp1 = self.gp1(self.mesh, x)
        
        vert_features_from_conv128 = vert_align(conv128, self.mesh) #TODO change the self.mesh to the output of the pooling.
        
        residual2_1 = x = self.activation(self.gc2_1(vert_features_from_conv128, mesh.edges_packed())) #TODO here the features are to be concatenated
        
        for i, l in enumerate(self.gc_module_2):
            x = self.activation(self.gc_module_2[i](x, mesh.edges_packed()))
            
        x = residual2_1 + x
            
        vertices = self.activation(self.gc2_14(x, mesh.edges_packed()))
        
        mesh_gc_2 = self.mesh = Meshes(verts=[vertices], faces=mesh.faces_list())
        
        self.mesh = self.gp2(self.mesh)
        
        vert_features_from_conv256 = vert_align(conv256, mesh) #TODO change the self.mesh to the output of the pooling.
        
        residual3_1 = x = self.activation(self.gc3_1(vert_features_from_conv256[0], mesh.edges_packed())) #TODO here the features are to be concatenated
        
        for i, l in enumerate(self.gc_module_3):
            x = self.activation(self.gc_module_3[i](x, mesh.edges_packed()))
            
        x = residual3_1 + x
        
        x = self.activation(self.gc3_14(x, mesh.edges_packed()))
        
        mesh_gc_3 = self.mesh = Meshes(verts=[x], faces=mesh.faces_list())
        
        return mesh_gc_1, mesh_gc_2, mesh_gc_3
    
if __name__ == "__main__":
    """Network Testing
    """
    import torch
    from pytorch3d.structures.meshes import Meshes
    from pytorch3d.utils import ico_sphere
    from pytorch3d.structures import join_meshes_as_batch
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    ico = ico_sphere(3).to(device=device)
    #print("Verts list", ico.verts_list())
    #print("Faces list", ico.faces_list())
    #print("Verts packed", ico.verts_packed())
    
    mesh_batch = join_meshes_as_batch([ico, ico])
    
    conv64 = torch.rand([2, 64, 54, 54], dtype=torch.float, device=device)
    conv128 = torch.rand([2, 128, 26, 26], dtype=torch.float, device=device)
    conv256 = torch.rand([2, 256, 12, 12], dtype=torch.float, device=device)
    conv512 = torch.rand([2, 512, 3, 3], dtype=torch.float, device=device)
    
    gcn = GraphConvolution(input_dim=64, hidden_dim=256 , output_dim = 3) # Need to understand the input dimension and the internal dimension how work
    gcn = gcn.to(device=device) # Move on GPU
    #print(GraphConvolution)
    output = gcn(mesh_batch, conv64, conv128, conv256)
    