from torch.nn import Module
from basenetwork.convolutionalnetwork import Convolutional
from basenetwork.graphconvolutionalnetwork import GraphConvolutional

class Pix2Mesh(Module):
    def __init__(self, image_input_channels: int = 3, convolutional_use_pooling: bool = False, convolutional_pooling_type: str = "Max", graph_convolutional_first_block_input_channels: int = 64, graph_convolutional_second_block_input_channels: int = 128, graph_convolutional_third_block_input_channels: int = 256, graph_convolutional_hidden_dim: int = 256, graph_convolutional_output_dim: int = 3, graph_convolutional_hidden_layer_count: int = 12):
        super(Pix2Mesh, self).__init__()
        
        self.convolutional_network = Convolutional(input_channel=image_input_channels, pooling=convolutional_use_pooling, pooling_type=convolutional_pooling_type)
        
        self.graph_convolutional_network = GraphConvolutional(first_block_input_dim=graph_convolutional_first_block_input_channels, second_block_input_dim=graph_convolutional_second_block_input_channels, third_block_input_dim=graph_convolutional_third_block_input_channels, hidden_dim=graph_convolutional_hidden_dim, output_dim=graph_convolutional_output_dim, hidden_layer_count=graph_convolutional_hidden_layer_count)
    
    def forward(self, image, mesh):
        conv16, conv32, conv64, conv128, conv256, conv512 = self.convolutional_network(image)
        
        """if len(conv16.shape()) == 4 or len(conv32.shape()) == 4 or len(conv64.shape()) == 4 or len(conv128.shape()) == 4 or len(conv256.shape()) == 4 or len(conv512.shape()) == 4:
            # Batch size different from 1
            for i in conv64.shape()[0]:"""
        mesh1, mesh2, mesh3 = self.graph_convolutional_network(mesh, conv64, conv128, conv256)
                
        return mesh1, mesh2, mesh3
        
        
        

if __name__ == "__main__":
    p2m = Pix2Mesh()
    p2m()