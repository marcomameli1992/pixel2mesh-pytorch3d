from torch import optim
from torch import nn
from torch.utils import data


def train(config, convolutional_model: nn.Module, graph_model: nn.Module, train_data: data.DataLoader, validation_data: data.DataLoader, epochs: int, running_save):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001) # See weight decay
    
    # define the optimizer
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            # get the input and labels 
            
            optimizer.zero_grad()
            
            # forward + backward + optimize
            conv16, conv32, conv64, conv128, conv256, conv512 = convolutional_model(data)
            
            if len(conv16.shape()) == 4 or len(conv32.shape()) == 4 or len(conv64.shape()) == 4 or len(conv128.shape()) == 4 or len(conv256.shape()) == 4 or len(conv512.shape()) == 4:
            # Batch size different from 1
                for i in conv64.shape()[0]:
                    mesh1, mesh2, mesh3 = graph_model(mesh, conv64, conv128, conv256) # TODO Define the data input
                    
                    