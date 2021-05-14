from __future__ import absolute_import
#import epdb
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Utils import center2corner
from .Config import Config

class smoothargmax2D(nn.Module):
    def __init__(self, template_size):
        super(smoothargmax2D, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        #Beta parameter controls impulse nature of softmax distribution
        self.beta = 5

    def forward(self, x):
        batchsz, h, w = x.size()

        # Projecting onto x,y axes by summing over other dimension
        x_w = x.sum(dim=1)
        x_h = x.sum(dim=2)

        # Impulse like softmax distribution
        probx_w = self.softmax(self.beta * x_w)
        probx_h = self.softmax(self.beta * x_h)
        
        # Indices along x,y axes
        idx_w = torch.arange(w, dtype=x.dtype, device = x.device).repeat(batchsz,1)
        idx_h = torch.arange(h, dtype=x.dtype, device = x.device).repeat(batchsz,1)

        # Compute smooth armax along x, y axes
        btid = torch.arange(batchsz, dtype=x.dtype, device = x.device, requires_grad=False).view(batchsz,1)
        maxY = (probx_w*idx_h).sum(dim=1).view(batchsz,1)
        maxX = (probx_w*idx_w).sum(dim=1).view(batchsz,1)
        lenH = template_size * torch.ones(batchsz, dtype=x.dtype, device = x.device, requires_grad=False).view(batchsz,1)
        lenW = template_size * torch.ones(batchsz, dtype=x.dtype, device = x.device, requires_grad=False).view(batchsz,1)
        
        return torch.cat([btid, maxY, maxX, lenH, lenW], dim=1)


class AltoG(nn.Module):    
    def __init__(self, config):
        super(AltoG, self).__init__()
        self.config = config
        alpha = 0.2

        # architecture (AlexNet like)
        self.feat_extraction = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 96, 11, 2)),                 # conv1
                ('bn1', nn.BatchNorm2d(96)),
                #('relu1', nn.ReLU(inplace=True)),
                ('lrelu1', nn.LeakyReLU(alpha, inplace=True) ),

                ('pool1', nn.MaxPool2d(3, 2)),

                ('conv2', nn.Conv2d(96, 256, 5, 1, groups=2)),      # conv2
                ('bn2', nn.BatchNorm2d(256)),
                #('relu2', nn.ReLU(inplace=True)),
                ('lrelu2', nn.LeakyReLU(alpha, inplace=True) ),

                ('pool2', nn.MaxPool2d(3, 2)),

                ('conv3', nn.Conv2d(256, 384, 3, 1)),               # conv3
                ('bn3', nn.BatchNorm2d(384)),
                #('relu3', nn.ReLU(inplace=True)),
                ('lrelu3', nn.LeakyReLU(alpha, inplace=True) ),

                ('conv4', nn.Conv2d(384, 384, 3, 1, groups=2)),     # conv4
                ('bn4', nn.BatchNorm2d(384)),
                #('relu4', nn.ReLU(inplace=True)),
                ('lrelu4', nn.LeakyReLU(alpha, inplace=True) ),

                ('conv5', nn.Conv2d(384, 256, 3, 1, groups=2))      # conv5
            ])
        )
 
        self.adjust = nn.Conv2d(1,1,1,1)
        self.softargmax = smoothargmax2D(config.template_size)
        self.initialize_weight()

    def forward(self, z, x):
        # get features for z and x
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)

        # correlation of z and x
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)
        bbox = self.softargmax(score.squeeze())
        center2corner()
        return score, bbox

    def xcorr(self, z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))
        return torch.cat(out, dim=0)

    def initialize_weight(self):
        for m in self.feat_extraction.modules():
            if isinstance(m, nn.Conv2d):
                    if self.config.initG == 'kaiming':
                        nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                        m.bias.data.fill_(.1)

                    elif self.config.initG == 'truncated':
                        def truncated_norm_init(data, stddev=0.01):
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

class AltoD(nn.Module):
    def __init__(self,config):
        super(AltoD, self).__init__()
        self.config = config
        self.discriminator = nn.Sequential(
            
            OrderedDict([
	            # 3x127x127 
	            ('conv1', nn.Conv2d(3, 16, 3, 2, bias=True) ),
	            ('bn1',	nn.InstanceNorm2d(16) ),
	            ('lrelu1', nn.LeakyReLU(0.2,inplace=True) ),

                # 64x63x63
                ('pool1', nn.MaxPool2d(3, 2)),
	            
	            # 64x31x31
	            ('conv2', nn.Conv2d(16, 32, 3, 2, bias=True) ),
	            ('bn2',	nn.InstanceNorm2d(32) ),
	            ('lrelu2', nn.LeakyReLU(0.2, inplace=True) ),

                # 128x15x15
                ('pool2', nn.MaxPool2d(3, 2)),
	            
	            # 128x7x7 
	            ('conv3', nn.Conv2d(32, 32, 3, 1, bias=True) ),
	            ('bn3',	nn.InstanceNorm2d(64) ),
	            ('lrelu3', nn.LeakyReLU(0.2,inplace=True) ),
	            
	            # 256x5x5 
	            ('conv4', nn.Conv2d(32, 64, 3, 1, bias=True) ),
	            ('bn4',	nn.InstanceNorm2d(128) ),
	            ('lrelu4', nn.LeakyReLU(0.2,inplace=True) ),
	            
	            # 512x3x3 
	            ('conv5', nn.Conv2d(64, 1, 3, 1, bias=True) ),
	            ('sig1', nn.Sigmoid() )
            ])
        )
        self.initialize_weight() 
        
    def forward(self, inputs):
        return self.discriminator(inputs)
    
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming initialization
                if self.config.initD == 'kaiming':
                    nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                    m.bias.data.fill_(.1)

                # truncated gaussian initialization
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
class lossobject():
    def BCEloss(self, prediction, label, weight):
        # weighted BCELoss 
        return F.binary_cross_entropy(prediction, label, weight, reduction='mean')

    def MSEloss(self, prediction, label):
        return F.mse_loss(prediction, label, reduction='mean')

    def Hingeloss(self, prediction, label):
        # HingeEmbeddingLoss
        return F.hinge_embedding_loss(prediction, label, margin=1.0, reduction='mean')

    def KLdivloss(self, prediction, label):
        # Kullback-Leibler divergence Loss.
        return F.kl_div(prediction, label, reduction='batchmean')

    def BCElogitsloss(self, prediction, label, weight):
        # weighted sigmoid cross entropy loss
        return F.binary_cross_entropy_with_logits(prediction, label, weight, reduction='mean')

    def customloss(self, prediction, label, weight):
        score, y, weights = prediction, label, weight
        a = -(score * y)
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a-b))
        loss = torch.mean(weights * loss)
        return loss


