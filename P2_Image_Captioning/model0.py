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
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    def forward(self, features, captions):
        ## features [10, 256] to [10, 1, 256]
        ## captions [10, 13] to [10, 12] ;  drop '<end>'
        ## embed(captions[:, :-1]) [10, 12, 256]
        ## torch.cat : stack at dim 1
        inputs_ = torch.cat((features.unsqueeze(1), self.embed(captions[:, :-1])), 1)
        out, self.hidden = self.lstm(inputs_)
        out = self.dropout(out)
        out = self.linear(out)
        # print(features.size())
        # print(self.embed(captions).size())
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ## inputs.size [1, 1, 256]
        result_caption = []
        for i in range(max_len):
            # print(inputs.size())
            out, states = self.lstm(inputs, states) ## [1, 1, 512]
            ##print(out.size())
            ##out = self.dropout(out.squeeze(1)) ## [1, 8855]
            out = self.linear(out.squeeze(1)) ## [1, 8855]
            ##print(out.size())
            target_idx = out.max(1)[1] ## out_max(1) [0] is value, [1] is index
            result_caption.append(target_idx.item())
            inputs = self.embed(target_idx).unsqueeze(1)
        return result_caption
