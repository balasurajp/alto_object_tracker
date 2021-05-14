'''AUTHORS : SURAJ & NAVEEN :)'''
import __init_paths
import os
import numpy as np
#from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.ops import roi_pool

from lib.VIDDataset import VIDDataset
from lib.DataAugmentation import RandomStretch, CenterCrop, RandomCrop, ToTensor
from lib.Utils import groundTruth, AverageMeter, center2corner, viewimages
from lib.eval_utils import centerThrErr

from models.Config import Config
from models.network import AltoG
from models.network import AltoD
from models.network import lossobject

def prepareData(config):
    # Data augmentation
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride
    
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.template_size, config.template_size)),
        ToTensor(),
    ])
    train_x_transforms = transforms.Compose([
        CenterCrop((config.instance_size, config.instance_size)),
        ToTensor(),
    ])

    valid_z_transforms = transforms.Compose([
        CenterCrop((config.template_size, config.template_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        CenterCrop((config.instance_size, config.instance_size)),
        ToTensor(),
    ])
    
    # Prepare Datasets
    trn_transforms = [train_z_transforms, train_x_transforms]
    val_transforms = [valid_z_transforms, valid_x_transforms]
    trn_dataset = VIDDataset(config, trn_transforms, mode="Train")
    val_dataset = VIDDataset(config, val_transforms, mode="Validation")

    # Create Dataloader
    trnloader = DataLoader(trn_dataset, batch_size=config.batchsize, shuffle=True,
                            num_workers=config.trn_numworkers, drop_last=True)

    valloader = DataLoader(val_dataset, batch_size=config.batchsize, shuffle=True,
                            num_workers=config.val_numworkers, drop_last=True)

    return trnloader, valloader

def saveSnapshot(snappath, net, optimizer, model):
    os.makedirs(snappath, exist_ok=True)
    netpath = os.path.join(snappath, 'net' + model +'_' + str(config.start_epoch) + '.pth')
    optpath = os.path.join(snappath, 'optimizer'+ model +'.pth') 
    
    torch.save(net.state_dict(), netpath)
    torch.save(optimizer.state_dict(), optpath)

def train(snapshotpath, config, device):
    #Tensor = torch.cuda.FloatTensor if config.use_gpu else torch.FloatTensor
    netG = AltoG()
    netD = AltoD()
    netG.load_state_dict(torch.load(config.pretrainedpath))
    if config.use_gpu:
        netG.to(device)
        netD.to(device)

    optimizerD = torch.optim.Adam(params=netD.parameters(), lr=config.lr, betas=config.betas)
    optimizerG = torch.optim.Adam(params=netG.parameters(), lr=config.lr, betas=config.betas)

    netG.train()
    netD.train()

    lossG = AverageMeter()
    lossD = AverageMeter()
    metric = AverageMeter()

    for i in range(config.start_epoch, config.num_epoch):
        # ========================= training ==========================
        lossG.reset()
        lossD.reset()
        metric.reset()

        for j, data in enumerate(trnloader):
            template_imgs, instance_imgs= data
            gtresps, _ = groundTruth((config.score_size, config.score_size), config )
            if config.use_gpu:
                template_imgs = template_imgs.cuda()
                instance_imgs = instance_imgs.cuda()

            # Adversarial ground labels
            rl = np.random.uniform(0.90,1.00)
            fk = np.random.uniform(0.00,0.10)
            real = torch.full((template_imgs.size(0),1,1,1), rl).cuda()
            fake = torch.full((template_imgs.size(0),1,1,1), fk).cuda()

            # ==============================================Train Discriminator====================================================
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            gnresps, bbox = netG(template_imgs, instance_imgs)
            bbox[:,1], bbox[:,2], bbox[:,3], bbox[:,4]  = center2corner( bbox[:,1], bbox[:,2], bbox[:,3], bbox[:,4] )
            roipatches = roi_pool(instance_imgs, bbox, (config.template_size, config.template_size))
            
            # DLoss
            if config.loss == "logistic":
                real_loss = BCEloss(netD(template_imgs), real)
                fake_loss = BCEloss(netD(roipatches.detach()), fake)
                dloss = (real_loss + fake_loss)/2
            dloss.backward()
            optimizerD.step()

            # ================================================Train Generator=======================================================
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            gnresps, bbox = netG(template_imgs, instance_imgs)
            gn_xy = bbox[:,1:3]
            gt_xy = 8.0 * torch.ones(gn_xy.size(), dtype = gn_xy.dtype).cuda()

            bbox[:,1], bbox[:,2], bbox[:,3], bbox[:,4]  = center2corner( bbox[:,1], bbox[:,2], bbox[:,3], bbox[:,4] )
            roipatches = roi_pool(instance_imgs, bbox, (config.template_size, config.template_size))

            # GLoss
            if config.loss == "logistic":
                gloss = 0.7*BCEloss(netD(roipatches), real) + 0.3*MSEloss(gn_xy, gt_xy)

            before = torch.empty_like( netG.feat_extraction.conv1.state_dict()["weight"] ).copy_( netG.feat_extraction.conv1.state_dict()["weight"] )
            gloss.backward()
            optimizerG.step()
            after = torch.empty_like( netG.feat_extraction.conv1.state_dict()["weight"] ).copy_( netG.feat_extraction.conv1.state_dict()["weight"] )
            if( torch.equal(before, after) ):
                print('model is not training')

            # Training Statistics
            lossG.update(gloss.data.cpu().numpy(), config.batchsize)
            lossD.update(dloss.data.cpu().numpy(), config.batchsize)  
            err = centerThrErr(gnresps.data.cpu().numpy(), gtresps.cpu().numpy())
            metric.update(err, config.batchsize)

            # Print Information
            if (j + 1) % config.logfreq == 0:
                print( "Epoch %d/%d (%d/%d) ||\t D loss: %f |\t G loss: %f |\t ErrorDisp: %f"
                        % (i+1, config.num_epoch, (j+1)*config.batchsize, config.numpairs, lossD.avg , lossG.avg, metric.avg) )

        if(i+1!=config.numepoch):
            schedulerG.step()
            schedulerD.step()
        
        saveSnapshot(config.snappath, netG, optimizerG, 'G')
        saveSnapshot(config.snappath, netD, optimizerD, 'D')
if __name__ == "__main__":
    config = Config()
    snappath = os.path.join(config.save_basepath, config.save_subpath)
    cwdpath = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
    config.cwd = cwdpath
    
    np.random.seed(1234)
    torch.manual_seed(1234)

    assert config is not None
    if config.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpuid)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Training ALTO
    trnloader, valloader = prepareData(config)


    train(config=config, device=device)