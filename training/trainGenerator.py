from __future__ import print_function
import initpaths

import torch

from models import generator
from handleData import DataGenerator


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

trngen = DataGenerator(datapath=config['trndata'], metadatapath=config['trnmetadata'], batchsize=config['batchsize'])
valgen = DataGenerator(datapath=config['valdata'], metadatapath=config['valmetadata'], batchsize=config['batchsize'])

