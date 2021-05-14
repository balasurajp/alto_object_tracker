from __future__ import absolute_import

import epdb
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Config import Config


class SiamNet_G(nn.Module):    
    def __init__(self):
        super(SiamNet_G, self).__init__()
        self.config = Config()

        # architecture (AlexNet like)
        self.feat_extraction = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 96, 11, 2)),             # conv1
                ('bn1', nn.BatchNorm2d(96)),
                #('relu1', nn.ReLU(inplace=True)),
                ('lrelu1', nn.LeakyReLU(0.01, inplace=True) ),

                ('pool1', nn.MaxPool2d(3, 2)),

                ('conv2', nn.Conv2d(96, 256, 5, 1, groups=2)),  # conv2
                ('bn2', nn.BatchNorm2d(256)),
                #('relu2', nn.ReLU(inplace=True)),
                ('lrelu2', nn.LeakyReLU(0.01, inplace=True) ),

                ('pool2', nn.MaxPool2d(3, 2)),

                ('conv3', nn.Conv2d(256, 384, 3, 1)),           # conv3
                ('bn3', nn.BatchNorm2d(384)),
                #('relu3', nn.ReLU(inplace=True)),
                ('lrelu3', nn.LeakyReLU(0.01, inplace=True) ),

                ('conv4', nn.Conv2d(384, 384, 3, 1, groups=2)),  # conv4
                ('bn4', nn.BatchNorm2d(384)),
                #('relu4', nn.ReLU(inplace=True)),
                ('lrelu4', nn.LeakyReLU(0.01, inplace=True) ),

                ('conv5', nn.Conv2d(384, 256, 3, 1, groups=2))  # conv5
            ])
        )

        # adjust layer 
        self.adjust = nn.Conv2d(1,1,1,1)
        # initialize weights
        self.initialize_weight()

    def forward(self, z, x):
        # get features for z and x
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)

        # correlation of z and x
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)

        return score

    def xcorr(self, z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))
        
        return torch.cat(out, dim=0)

    def initialize_weight(self):
        #initialize network parameters
        for m in self.feat_extraction.modules():
            if isinstance(m, nn.Conv2d):
                    # kaiming initialization
                    if self.config.initG == 'kaiming':
                        nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                        m.bias.data.fill_(.1)

                    elif self.config.initG == 'truncated':
                        def truncated_norm_init(data, stddev=.01):
                            weight = np.random.normal(size=data.shape)
                            weight = np.clip(weight,
                                             a_min=-2*stddev, a_max=2*stddev)
                            weight = torch.from_numpy(weight).float()
                            return weight
                        m.weight.data = truncated_norm_init(m.weight.data)
                        m.bias.data.fill_(.1)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.momentum = self.config.bn_momentum

        self.adjust.weight.data.fill_(1e-3)
        self.adjust.bias.data.zero_()

########################################################################

class SiamNet_D(nn.Module):
    def __init__(self):
        super(SiamNet_D, self).__init__()
        self.config = Config()
        self.discriminator = nn.Sequential(
            
            OrderedDict([
	            # 3x127x127 - (3,64,3,2)
	            ('conv1', nn.Conv2d(3, 64, 3, 2, bias=True) ),
	            ('bn1',	nn.BatchNorm2d(64) ),
	            ('lrelu1', nn.LeakyReLU(0.2,inplace=True) ),

                # 64x63x63 - (64,64,3,2)
                ('pool1', nn.MaxPool2d(3, 2)),
	            
	            # 64x31x31 - (64,128,3,2)
	            ('conv2', nn.Conv2d(64, 128, 3, 2, bias=True) ),
	            ('bn2',	nn.BatchNorm2d(128) ),
	            ('lrelu2', nn.LeakyReLU(0.2, inplace=True) ),

                # 128x15x15 - (128,128,3,2)
                ('pool2', nn.MaxPool2d(3, 2)),
	            
	            # 128x7x7 - (128,256,3,1)
	            ('conv3', nn.Conv2d(128, 256, 3, 1, bias=True) ),
	            ('bn3',	nn.BatchNorm2d(256) ),
	            ('lrelu3', nn.LeakyReLU(0.2,inplace=True) ),
	            
	            # 256x5x5 - (256,512,3,1)
	            ('conv4', nn.Conv2d(256, 512, 3, 1, bias=True) ),
	            ('bn4',	nn.BatchNorm2d(512) ),
	            ('lrelu4', nn.LeakyReLU(0.2,inplace=True) ),
	            
	            # 512x3x3 - (512,1,3,1)
	            ('conv5', nn.Conv2d(512, 1, 3, 1, bias=True) ),
	            ('sig1', nn.Sigmoid() )
            ])
        )
        
        # initialize weights
        self.initialize_weight() 
        
        
    def forward(self, inputs):
        return self.discriminator(inputs)
    
    def initialize_weight(self):
        """initialize network parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming initialization
                if self.config.initD == 'kaiming':
                    nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                    m.bias.data.fill_(.1)

                elif self.config.initD == 'truncated':
                    def truncated_norm_init(data, stddev=.01):
                        weight = np.random.normal(size=data.shape)
                        weight = np.clip(weight,
                                         a_min=-2*stddev, a_max=2*stddev)
                        weight = torch.from_numpy(weight).float()
                        return weight
                    m.weight.data = truncated_norm_init(m.weight.data)
                    m.bias.data.fill_(.1)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.momentum = self.config.bn_momentum

#######################################################################################
class lossfn():
    def adversarial_loss(self, prediction, label, weight):
        # weighted BCELoss 
        return F.binary_cross_entropy(prediction, label, weight, reduction='mean')

    def mse_loss(self, prediction, label):
        return F.mse_loss(prediction, label, reduction='mean')

    def hinge_loss(self, prediction, label):
        # HingeEmbeddingLoss
        return F.hinge_embedding_loss(prediction, label, margin=1.0, reduction='mean')

    def kldiv_loss(self, prediction, label):
        # Kullback-Leibler divergence Loss.
        return F.kl_div(prediction, label, reduction='batchmean')

    def weight_loss(self, prediction, label, weight):
        # weighted sigmoid cross entropy loss
        return F.binary_cross_entropy_with_logits(prediction, label, weight, reduction='mean')

    def customize_loss(self, prediction, label, weight):
        score, y, weights = prediction, label, weight
        a = -(score * y)
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a-b))
        loss = torch.mean(weights * loss)
        return loss


