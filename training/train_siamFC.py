'''AUTHORS: SURAJ & NAVEEN :)'''
import __init_paths

import os
import numpy as np
#from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lib.VIDDataset import VIDDataset
from lib.DataAugmentation import RandomStretch, CenterCrop, RandomCrop, ToTensor
from lib.Utils import create_label, extract_Dpatches, viewimages, AverageMeter
from lib.eval_utils import centerThrErr

from models.Config import Config
from models.network import SiamNet_G
from models.network import SiamNet_D
from models.network import lossfn

np.random.seed(1357)
torch.manual_seed(1234)

#IN_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

def netParams(netG,netD):
    # ===================Generator============================
    paramsG = []
    for key, value in dict(netG.feat_extraction.named_parameters()).items():
        if 'conv' in key:
            if 'bias' in key:
                paramsG += [{'params': [value], 'weight_decay': 0}]
            else:   # weight
                paramsG += [{'params': [value], 'weight_decay': config.weightdecay}]
        if 'bn' in key:
            paramsG += [{'params': [value], 'weight_decay': 0}]

    paramsG += [{'params': [netG.adjust.bias]}, {'params': [netG.adjust.weight]}]

    # =================Discriminator=========================
    paramsD = []
    for key, value in dict(netD.discriminator.named_parameters()).items():
        if 'conv' in key:
            if 'bias' in key:
                paramsD += [{'params': [value], 'weight_decay': 0}]
            else:   # weight
                paramsD += [{'params': [value], 'weight_decay': config.weightdecay}]
        if 'bn' in key:
            paramsD += [{'params': [value], 'weight_decay': 0}]

    return paramsG, paramsD

def netOptimizer(net, params, snap_path, config, flag):
    if (flag=='D'):
        optimizer = torch.optim.SGD(params=params, lr=config.lr, momentum=config.momentum, weight_decay=config.weightdecay)
    else:
        optimizer = torch.optim.Adam(params=params, lr=config.lr, betas=config.betas, weight_decay=config.weightdecay)

    # LR Scheduler
    if not config.resume:
        trainLR = np.logspace(-2, -5, config.numepoch)
        scheduler = LambdaLR(optimizer, lambda epoch: trainLR[epoch])
    
    else:
        trainLR = np.logspace(-2, -5, config.numepoch)
        trainLR = trainLR[config.start_epoch:]

        netpath = os.path.join(snap_path, 'net' + flag +'_' + str(config.start_epoch) + '.pth')
        optpath = os.path.join(snap_path, 'optimizer'+ flag +'.pth')
        net.load_state_dict(torch.load(savepath))
        optimizer.load_state_dict(torch.load(optpath))
        
        scheduler = LambdaLR(optimizer, lambda epoch: trainLR[epoch])
        print('Resume training from epoch {}'.format(config.start_epoch))
    return optimizer, scheduler

def saveSnapshot(snap_path, net, optimizer, flag):
    os.makedirs(snap_path, exist_ok=True)
    netpath = os.path.join(snap_path, 'net' + flag +'_' + str(config.start_epoch) + '.pth')
    optpath = os.path.join(snap_path, 'optimizer'+ flag +'.pth') 
    
    torch.save(net.state_dict(), netpath)
    torch.save(optimizer.state_dict(), optpath)

def train(nets, optimizers, schedulers, dataloaders, lossfunction, snap_path, config):
    Tensor = torch.cuda.FloatTensor if config.use_gpu else torch.FloatTensor
    
    netG, netD = nets
    optimizerG, optimizerD = optimizers
    schedulerG, schedulerD = schedulers
    trnloader, valloader = dataloaders
    
    gtresps, resp_weights = create_label( [config.score_size, config.score_size], config)
    for i in range(config.start_epoch, config.numepoch):
        # ========================= training ==========================
        netG.train()
        netD.train()
    
        lossG = AverageMeter()
        lossD = AverageMeter()
        metric = AverageMeter()

        for j, data in enumerate(trnloader):
            template_imgs, instance_imgs= data
            if config.use_gpu:
                template_imgs = template_imgs.cuda()
                instance_imgs = instance_imgs.cuda()

            # Adversarial ground labels
            rl = np.random.uniform(0.85,1.00)
            fk = np.random.uniform(0.00,0.15)
            real = Variable(Tensor(template_imgs.size(0), 1, 1, 1).fill_(rl), requires_grad=False)
            fake = Variable(Tensor(template_imgs.size(0), 1, 1, 1).fill_(fk), requires_grad=False)

            # ==============================================Train Discriminator====================================================
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            gnresps = netG(Variable(template_imgs), Variable(instance_imgs))
            gn_patches, bx_gn, gt_patches, bx_gt = extract_Dpatches(instance_imgs, gnresps)
            gn_patches = Variable(gn_patches, requires_grad=True)
            bx_gn = Variable(bx_gn, requires_grad=True)
            
            # DLoss
            if config.loss == "logistic":
                real_loss = lossfun.adversarial_loss(netD(gt_patches), real, weight = None)
                fake_loss = lossfun.adversarial_loss(netD(gn_patches.detach()), fake, weight = None)
                dloss = (real_loss + fake_loss)/2
            dloss.backward()
            optimizerD.step()

            # ================================================Train Generator=======================================================
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            gnresps = netG(Variable(template_imgs), Variable(instance_imgs))
            gn_patches, bx_gn, gt_patches, bx_gt = extract_Dpatches(instance_imgs, gnresps)
            gn_patches = Variable(gn_patches, requires_grad=True)
            bx_gn = Variable(bx_gn, requires_grad=True)

            # GLoss
            if config.loss == "logistic":
                gloss = 0.7*lossfun.adversarial_loss(netD(gn_patches), real, weight=None) + 0.3*lossfun.mse_loss(bx_gn,bx_gt)

            print('hello', netG.feat_extraction.conv1.state_dict()["weight"])
            gloss.backward()
            optimizerG.step()
            print('world', netG.feat_extraction.conv1.state_dict()["weight"])
            
            # Training Statistics
            lossG.update(gloss.data.cpu().numpy(), config.batchsize)
            lossD.update(dloss.data.cpu().numpy(), config.batchsize)  
            err = centerThrErr(gnresps.data.cpu().numpy(), gtresps.cpu().numpy())
            metric.update(err, config.batchsize)

            # Print Information
            if (j + 1) % config.logfreq == 0:
                print( "Epoch %d/%d (%d/%d) ||\t D loss: %f |\t G loss: %f |\t ErrorDisp: %f"
                        % (i+1, config.numepoch, (j+1)*config.batchsize, config.numpairs, lossD.avg , lossG.avg, metric.avg) )

        if(i+1!=config.numepoch):
            schedulerG.step()
            schedulerD.step()
        
        saveSnapshot(snap_path, netG, optimizerG, 'G')
        saveSnapshot(snap_path, netD, optimizerD, 'D')
if __name__ == "__main__":
    config = Config()
    snappath = os.path.join(config.save_basepath, config.save_subpath)
    
    assert config is not None
    if config.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpuid)

    netG = SiamNet_G()
    netG.load_state_dict(torch.load(config.pretrained_modelpath))
    netD = SiamNet_D()
    lossfun = lossfn()
    if config.use_gpu:
        netG.cuda()
        netD.cuda()

    # Training logistics
    paramsG, paramsD = netParams(netG,netD)
    optimizerG, schedulerG = netOptimizer(netG, paramsG, snappath, config, 'G')
    optimizerD, schedulerD = netOptimizer(netD, paramsG, snappath, config, 'D')
    
    nets = (netG, netD)
    optimizers =(optimizerG, optimizerD)
    schedulers =(schedulerG, schedulerD)
    dataloaders = prepareData(config)

    # Training SiamFC network
    train(nets, optimizers, schedulers, dataloaders, lossfun, snap_path = snappath, config=config)