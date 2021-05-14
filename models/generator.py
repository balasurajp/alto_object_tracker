from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class generator(nn.Module):
    def __init__(self, inputChannels):
        super(generator, self).__init__()
        self.inputChannels = inputChannels

        # Encoder layers
        self.econv1_0 = nn.Sequential(*[ nn.Conv2d(in_channels=self.inputChannels, out_channels=64,
                                                  kernel_size=5),
                                         nn.BatchNorm2d(64) ])
        self.econv1_1 = nn.Sequential(*[ nn.Conv2d(in_channels=64, out_channels=64,
                                                  kernel_size=5),
                                         nn.BatchNorm2d(64) ])

        self.econv2_0 = nn.Sequential(*[ nn.Conv2d(in_channels=64, out_channels=128,
                                                  kernel_size=3),
                                         nn.BatchNorm2d(128) ])
        self.econv2_1 = nn.Sequential(*[ nn.Conv2d(in_channels=128, out_channels=128,
                                                  kernel_size=3),
                                         nn.BatchNorm2d(128) ])

        self.econv3_0 = nn.Sequential(*[ nn.Conv2d(in_channels=128, out_channels=256,
                                                  kernel_size=3),
                                         nn.BatchNorm2d(256) ])
        self.econv3_1 = nn.Sequential(*[ nn.Conv2d(in_channels=256, out_channels=256,
                                                  kernel_size=3),
                                         nn.BatchNorm2d(256) ])

        # Decoder layers
        self.dconvtr1_0 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                             kernel_size=3),
                                          nn.BatchNorm2d(256) ])
        self.dconvtr1_1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                                             kernel_size=3),
                                          nn.BatchNorm2d(128) ])

        self.dconvtr2_0 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                                             kernel_size=3),
                                          nn.BatchNorm2d(128) ])
        self.dconvtr2_1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                                             kernel_size=3),
                                          nn.BatchNorm2d(64) ])

        self.dconvtr3_0 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                                             kernel_size=3),
                                          nn.BatchNorm2d(64) ])
        self.dconvtr3_1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                                             kernel_size=3),
                                          nn.BatchNorm2d(64) ])

        self.apply(self.initializeWeights)

    def getEmbedding(self, x):
    	# Encoder
        L1 = inputImg.size()
        x1_0 = F.relu(self.econv1_0(x))
        x1_1 = F.relu(self.econv1_1(x1_0))
        x1, indices1 = F.max_pool2d(x1_1, kernel_size=3, stride=2, return_indices=True)

        L2 = x1.size()
        x2_0 = F.relu(self.econv2_0(x1))
        x2_1 = F.relu(self.econv2_1(x2_0))
        x2, indices2 = F.max_pool2d(x2_1, kernel_size=3, stride=2, return_indices=True)

        L3 = x2.size()
        x3_0 = F.relu(self.econv3_0(x2))
        x3_1 = F.relu(self.econv3_1(x3_0))
        x3, indices3 = F.max_pool2d(x3_1, kernel_size=3, stride=2, return_indices=True)

        # Decoder
        x3_0d = F.max_unpool2d(x3, indices3, kernel_size=3, stride=2, output_size=L3)
        x3_1d = F.relu(self.dconvtr1_0(x3_0d))
        x3_2d = F.relu(self.dconvtr1_1(x3_1d))

        x2_0d = F.max_unpool2d(x2, indices2, kernel_size=3, stride=2, output_size=L2)
        x2_1d = F.relu(self.dconvtr2_0(x2_0d))
        x2_2d = F.relu(self.dconvtr2_1(x2_1d))

        x1_0d = F.max_unpool2d(x1, indices1, kernel_size=3, stride=2, output_size=L1)
        x1_1d = F.relu(self.dconvtr1_0(x1_0d))
        x1_2d = F.relu(self.dconvtr1_1(x1_1d))
        return x1_2d

    def crossCorr(self, z, x):
        responses = []
        for i in range(x.size(0)):
            responses.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))
        return torch.cat(responses, dim=0)

    def forward(self, zImgs, xImgs):
        zEmbeds = self.getEmbedding(zImgs)
        xEmbeds = self.getEmbedding(xImgs)

        response = self.crossCorr(zEmbeds, xEmbeds)
        return response

    def computeloss(self, pdresponse, gtresponse):
    	loss = F.binary_cross_entropy(pdresponse, gtresponse, weight=None, reduction='mean')
    	return loss

    def initializeWeights(self, m):
	    classname = m.__class__.__name__
	    if classname.find('Conv') != -1:
	        nn.init.normal_(m.weight.data, 0.0, 0.02)
	    elif classname.find('BatchNorm') != -1:
	        nn.init.normal_(m.weight.data, 1.0, 0.02)
	        nn.init.constant_(m.bias.data, 0)