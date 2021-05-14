import __init_paths

import os
import numpy as np
from tqdm import tqdm

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

'''
IN_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_image_transforms = transforms.Compose([
        ToTensor(),
        IN_normalize,
    ])
'''

def prepareData(config):
    # Data augmentation
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride
    
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    train_x_transforms = transforms.Compose([
        CenterCrop((config.instance_size, config.instance_size)),
        ToTensor(),
    ])

    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        CenterCrop((config.instance_size, config.instance_size)),
        ToTensor(),
    ])

    # Prepare Datasets
    train_dataset = VIDDataset(config.train_imdb, config.data_dir, config,
                               train_z_transforms, train_x_transforms)
    val_dataset = VIDDataset(config.val_imdb, config.data_dir, config, 
                             valid_z_transforms, valid_x_transforms, "Validation")

    # Create Dataloader
    trn_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.train_num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.val_num_workers,
                            drop_last=True)

    return trn_loader, val_loader

def netParams(netG,netD):
    # ===================Generator============================
    paramsG = []
    for key, value in dict(netG.feat_extraction.named_parameters()).items():
        if 'conv' in key:
            if 'bias' in key:
                paramsG += [{'params': [value],
                            'weight_decay': 0}]
            else:   # weight
                paramsG += [{'params': [value],
                            'weight_decay': config.weight_decay}]
        if 'bn' in key:
            paramsG += [{'params': [value],
                        'weight_decay': 0}]

    paramsG += [{'params': [netG.adjust.bias]},
                {'params': [netG.adjust.weight]}]

    # =================Discriminator=========================
    paramsD = []
    for key, value in dict(netD.discriminator.named_parameters()).items():
        if 'conv' in key:
            if 'bias' in key:
                paramsD += [{'params': [value],
                            'weight_decay': 0}]
            else:   # weight
                paramsD += [{'params': [value],
                            'weight_decay': config.weight_decay}]
        if 'bn' in key:
            paramsD += [{'params': [value],
                        'weight_decay': 0}]

    return paramsG, paramsD

def netOptimizer(net, params, snap_path, config, flag):
    optimizer = torch.optim.SGD(params,
                                config.lr,
                                config.momentum,
                                config.weight_decay)

    # Adjusting LR over epoches
    if not config.resume:
        train_lrs = np.logspace(-2, -5, config.num_epoch)
        scheduler = LambdaLR(optimizer, lambda epoch: train_lrs[epoch])
    
    else:
        train_lrs = np.logspace(-2, -5, config.num_epoch)
        train_lrs = train_lrs[config.start_epoch:]

        net.load_state_dict(torch.load(os.path.join(snap_path,
                                        'net'+ flag +'_' + str(config.start_epoch) +
                                        '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(snap_path,
                                             'optimizer'+ flag +'.pth')))
        
        scheduler = LambdaLR(optimizer, lambda epoch: train_lrs[epoch])
        print('Resume training from epoch {}'.format(config.start_epoch))

    return optimizer, scheduler

def saveSnapshot(snap_path, net, optimizer, flag):
    if not os.path.exists(snap_path):
        os.makedirs(snap_path)

    torch.save(net.state_dict(),
               os.path.join(snap_path,
                            'net'+ flag +'_' + str(config.start_epoch) + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(snap_path,
                            'optimizer'+ flag +'.pth'))

def train(netG, netD, lossfunction,
          optimizerG, optimizerD, schedulerG, schedulerD,
          train_loader, val_loader,
          snap_path=None, config=None):
    # CUDA Datatype
    Tensor = torch.cuda.FloatTensor if config.use_gpu else torch.FloatTensor
    
    gtresps, resp_weights = create_label( (config.score_size, config.score_size), config, config.use_gpu)
    
    # ===================== training & validation ========================
    for i in range(config.start_epoch, config.num_epoch):
        # adjusting learning rate
        schedulerG.step()
        schedulerD.step()

        # ========================= training ==========================
        netG.train()
        netD.train()
        
        # Loss meters and Acuuracy Metrics
        lossG = AverageMeter()
        lossD = AverageMeter()
        metric = 0
        samples = 0

        for j, data in enumerate(train_loader):
            exemplar_imgs, instance_imgs= data
            if config.use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()

            # Adversarial ground labels
            real = Variable(Tensor(exemplar_imgs.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(exemplar_imgs.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)
            # ====================Train Discriminator====================   
            optimizerD.zero_grad()

            gnresps = netG(Variable(exemplar_imgs), Variable(instance_imgs))
            gn_patches, bx_gn, gt_patches, bx_gt = extract_Dpatches(instance_imgs, gnresps)
            
            # DLoss
            if config.loss == "logistic":
                real_loss = lossfun.adversarial_loss(netD(gt_patches), real, weight = None)
                fake_loss = lossfun.adversarial_loss(netD(gn_patches.detach()), fake, weight = None)
                dloss = (real_loss + fake_loss)/2
            dloss.backward()
            optimizerD.step()

            # ===========================
            # Train Generator
            # ===========================
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            gnresps = netG(Variable(exemplar_imgs), Variable(instance_imgs))
            gn_patches, bx_gn, gt_patches, bx_gt = extract_Dpatches(instance_imgs, gnresps)

            # GLoss
            if config.loss == "logistic":
                gloss = 0.7*lossfun.adversarial_loss(netD(gn_patches), real, weight=None) + 0.3*lossfun.mse_loss(bx_gn,bx_gt)                
            gloss.backward()
            optimizerG.step()
            

            # Training Statistics
            lossG.update(gloss.data.cpu().numpy(), samples+config.batch_size)
            lossD.update(gloss.data.cpu().numpy(), samples+config.batch_size)  

            metric = centerThrErr(pdresps.data.cpu().numpy(),
                                  gtresps.cpu().numpy(),
                                  metric, samples)
            samples += config.batch_size

            # Print Information
            if (j + 1) % config.log_freq == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] || [D loss: %f] [G loss: %f] [ErrorDisp: %f]"
                    % (i+1, config.num_epoch, (j+1), config.num_pairs/config.batch_size,
                    lossD.avg , lossG.avg, metric))

        # ------------------------- saving model ---------------------------
        saveSnapshot(snap_path, netG, optimizerG, 'G')
        saveSnapshot(snap_path, netD, optimizerD, 'D')


if __name__ == "__main__":
    # initialize training configuration
    config = Config()
    snap_path   = os.path.join(config.save_base_path, config.save_sub_path)
    
    assert config is not None
    #gpus = [int(i) for i in config.gpu_id.split(',')]
    if config.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    # Prepare Data
    trn_loader, val_loader = prepareData(config)

    # ALTO Network and Optimization
    netG = SiamNet_G()
    netG.load_state_dict(torch.load(config.pretrained_model_path))
    netD = SiamNet_D()
    lossfun = lossfn()

    # Training logistics
    paramsG, paramsD = netParams(netG,netD)
    optimizerG, schedulerG = netOptimizer(netG, paramsG, snap_path, config, 'G')
    optimizerD, schedulerD = netOptimizer(netD, paramsG, snap_path, config, 'D')

    if config.use_gpu:
        #netG = torch.nn.DataParallel(netG, device_ids=gpus).cuda()
        #netD = torch.nn.DataParallel(netD, device_ids=gpus).cuda()
        netG.cuda()
        netD.cuda()

    # Training SiamFC network
    train(netG, netD, lossfun,
          optimizerG, optimizerD, schedulerG, schedulerD,
          trn_loader, val_loader,
          snap_path = snap_path, config=config)