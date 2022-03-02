import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class StructuredAutoencoderNet(nn.Module):
    ## Initialize the network
    def __init__(self, p, encoder_config, decoder_config, dropout_rate = 0):

        super().__init__()
        self.p = p
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        ## Save linear layer weights in a list
        self.weights_layer = []

        ## Generate encoder layer 
        index = 0
        self.encoder_layer = []
        for i in range(len(self.encoder_config['dimension']) - 1):
            self.encoder_layer.append(("linear" + str(index), nn.Linear(int(self.encoder_config['dimension'][i]), int(self.encoder_config['dimension'][i + 1]))))
            if i != len(self.encoder_config['dimension']) - 2:
                self.encoder_layer.append(("Sigmoid" + str(index), nn.Sigmoid()))
                self.encoder_layer.append(("dropout" + str(index), nn.Dropout(p = dropout_rate)))
            index += 1
        
        for index, layer in enumerate(self.encoder_layer):
            if layer[0] == "linear":
                self.weights_layer.append(torch.nn.Parameter(layer[1].weight))
                self.encoder_layer[index][1].weight = self.weights_layer[-1]

        ## Generate decoder layer
        index = 0
        self.decoder_layer = []

        for i in range(len(self.decoder_config['dimension']) - 1):
            # decoder_layer.append(("relu" + str(index),nn.ReLU()))
            if i != 0:
                self.decoder_layer.append(("dropout" + str(index), nn.Dropout(p = dropout_rate)))

            self.decoder_layer.append(("linear" + str(index), nn.Linear(int(self.decoder_config['dimension'][i]), int(self.decoder_config['dimension'][i + 1]))))
            if i != len(self.decoder_config['dimension']) - 2:
                self.decoder_layer.append(("Sigmoid" + str(index), nn.Sigmoid()))
            index += 1

        ## encoder_net and decoder_net
        self.encoder_net = nn.Sequential(OrderedDict(
          self.encoder_layer
        ))

        self.decoder_net = nn.Sequential(OrderedDict(
          self.decoder_layer
        ))
    
    # encode and decode function
    def encode(self, X, mask):
        index = 0
        for layer in self.encoder_layer:
            if layer[0] == "linear":
                X = torch.nn.functional.linear(X, self.weights_layer[index])
                index += 1
            else:
                X = layer[1](X)
        # for layer in self.encoder_net:
        #     X = layer(X)
        X = X * mask
        return X 

    def decode(self, X):
        index = len(self.weights_layer) - 1
        for layer in self.decoder_layer:
            if layer[0] == "linear":
                X = torch.nn.functional.linear(X, self.weights_layer[index].t())
                index -= 1
            else:
                X = layer[1](X)
        # for layer in self.decoder_net:
        #     X = layer(X)
        return X

    # forward network
    def forward(self, X, mask):
        X = self.encode(X, mask)
        X = self.decode(X)
        
        return X 

def StructuredMaskedAutoencoder(dataset, group, comp_num, N, p, encoder_dimension, decoder_dimension):
    model = StructuredAutoencoderNet(p, {'dimension' : encoder_dimension}, {'dimension' : decoder_dimension})
    print(model)
    
    # Training configuration setting
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())
        
    # Generate tensor list
    tensor_data_list = []
    for i in range(N):
        tensor_data_list.append(torch.tensor(dataset[i].astype(np.float32).T))
        
    # Generate tensor list for mask
    tensor_mask = []
    for i in range(N):
        tensor_mask.append([0 for i in range(int(sum(comp_num)))])
    mask_loc_index = 0
    # print(group)

    for i in range(len(group)):
        # Check type is list 
        if not isinstance(group[i], list):
            group[i] = [group[i]]

        for data_id in group[i]:
            for t in range(int(comp_num[i])):
                # print(int(data_id), mask_loc_index)
                tensor_mask[int(data_id) - 1][mask_loc_index + t] = 1
        mask_loc_index += int(comp_num[i])

    for index, mask in enumerate(tensor_mask):
        tensor_mask[index] = torch.tensor(mask)
    
    # Training process
    for epoch in range(100):
        tensor_pred_list = []
        for i in range(N):
            tensor_pred_list.append(model(tensor_data_list[i], tensor_mask[i]))
        
        loss = 0
        for i in range(N):
            loss += criterion(tensor_data_list[i], tensor_pred_list[i])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Embedding
    embedding_list = []
    for i in range(N):
        embedding_list.append(model.encode(tensor_data_list[i], tensor_mask[i]).detach().numpy())
    
    return embedding_list
    
