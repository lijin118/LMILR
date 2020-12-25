import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)


    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class encoder_template(nn.Module):

    def __init__(self,input_dim,latent_size,hidden_size_rule,device):
        super(encoder_template,self).__init__()


        # 'resnet_features': (1560, 1660),'attributes': (1450, 665)
        if len(hidden_size_rule)==2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule)==3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1] , latent_size]

        modules = []
        for i in range(len(self.layer_sizes)-2):

            modules.append(nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]))
            modules.append(nn.ReLU())

        self.feature_encoder = nn.Sequential(*modules)

        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)


        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)


        # self._lowdim = nn.Linear(in_features=self.layer_sizes[-2], out_features=16)
        # self._lowdim = nn.Linear(in_features=latent_size, out_features=16)


        self.apply(weights_init)

        self.to(device)


    def forward(self,x):
        h = self.feature_encoder(x)
        mu =  self._mu(h)
        logvar = self._logvar(h)
        # sigma = torch.exp(logvar)
        # eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        # eps = eps.expand(sigma.size())
        # z = mu + sigma * eps
        # # low_dim_z = self._lowdim(z)

        return mu, logvar

    # def forward_without_reparameter(self,x):
    #     h = self.feature_encoder(x)
    #     mu =  self._mu(h)
    #     return mu

class decoder_template(nn.Module):

    def __init__(self,input_dim,output_dim,hidden_size_rule,device):
        super(decoder_template,self).__init__()


        self.layer_sizes = [input_dim, hidden_size_rule[-1] , output_dim]

        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))

        self.apply(weights_init)

        self.to(device)
    def forward(self,x):

        return self.feature_decoder(x)
