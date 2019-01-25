import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size 
        self.num_layers = num_layers 

        # embedding layer that turns words into a vector of a specified size
        self.caption_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # the LSTM takes embedded caption vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # the linear layer that maps the hidden state output dimension 
        # to the number of vocabularies we want as output, vocab_size 
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.init_weights()
      
    
    def forward(self, features, captions):
        # create embedded caption vectors 
        embeds = self.caption_embeddings(captions[:, :-1])
        
        embeds = torch.cat([features.unsqueeze(1), embeds], dim=1)

        # get the output and hidden state by passing the lstm over our caption embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(embeds)
        
        out = self.fc(lstm_out)
        return out

                
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions = []

        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            out = self.fc(lstm_out)
            out = out.squeeze(1)                 
            _, predicted = torch.max(out, dim=1)
            captions.append(predicted.item())
            inputs = self.caption_embeddings(predicted)
            inputs = inputs.unsqueeze(1)

        return captions

