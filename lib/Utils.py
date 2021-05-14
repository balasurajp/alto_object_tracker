import torch
import numpy as np
import cv2
from torchvision import utils

'''
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
def corner2center(corner):
    """
    [x1, y1, x2, y2] --> [cx, cy, w, h]
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


Center = namedtuple('Center', 'x y w h')
'''
def roibbox(center, sidelens):
    batchsize, _ = center.size()
    btid = torch.arange(batchsize).view(batchsize,1) 
    lenh = torch.full((batchsize,1), sidelens[0])
    lenw = torch.full((batchsize,1), sidelens[1])

    centerbbox = torch.cat( (center[:,0], center[:1], lenh, lenw), dim=1)



def center2corner(center):
    # [cx, cy, w, h] --> [x1, y1, x2, y2]
    x, y, w, h = center[:,0], center[:,1], center[:,2], center[:,3]
    x1 = x - (w * 0.5)
    y1 = y - (h * 0.5)
    x2 = x + (w * 0.5)
    y2 = y + (h * 0.5)

    x1, y1, x2, y2 = x1.view(center.size(0), 1), y1.view(center.size(0), 1), x2.view(center.size(0), 1), y2.view(center.size(0), 1)
    center = torch.cat( (x1,y1,x2,y2), dim=1)
    return corner


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        
def viewimages(images):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Images")
    img = utils.make_grid(images, nrow=4, padding=2, normalize=False).cpu().numpy()
    img = np.asarray( img, dtype="uint8" )
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()

def extract_Dpatches(images, pdresps):
    # Extract patch from search image where correlation response is high
    gn_patches = []
    gt_patches = []
    #bx_gt = torch.zeros(images.size()[0], 2).cuda()
    #bx_gn = torch.zeros(images.size()[0], 2).cuda()

    bx_gn=[]
    bx_gt=[]
    
    for n, image, pdresp in zip(range( images.size()[0] ), images, pdresps):
        __,index = torch.max(pdresp.squeeze().view(-1), 0)

        i = index/pdresp.shape[1]
        j = index%pdresp.shape[2]
        
        rp = (63 + 8*i) 
        cp = (63 + 8*j)
        rg = image.shape[1]//2
        cg = image.shape[2]//2
            
        bx_gn.append( [rp,cp] )
        bx_gt.append( [rg,cg] )

        gn_patch = image[:, rp-63:rp+63+1, cp-63:cp+63+1]
        gt_patch = image[:, rg-63:rg+63+1, cg-63:cg+63+1]   
        gn_patches.append(gn_patch)
        gt_patches.append(gt_patch)
    
    return torch.stack(gn_patches),torch.cuda.FloatTensor(bx_gn) / images.size()[-1], torch.stack(gt_patches), torch.cuda.FloatTensor(bx_gt) / images.size()[-1]


def create_logisticloss_label(label_size, rPos, rNeg, neg_label=-1):
    """
    construct label for logistic loss (same for all pairs)
    """
    label_side = int(label_size[0])
    # logloss_label = torch.zeros(label_side, label_side)
    logloss_label = torch.from_numpy(np.full((label_side, label_side), neg_label)).float()
    label_origin = np.array([np.ceil(label_side / 2), np.ceil(label_side / 2)])
    for i in range(label_side):
        for j in range(label_side):
            dist_from_origin = np.sqrt((i - label_origin[0]) ** 2 + (j - label_origin[1]) ** 2)
            if dist_from_origin <= rPos:
                logloss_label[i, j] = +1
            else:
                if dist_from_origin <= rNeg:
                    logloss_label[i, j] = 0

    return logloss_label


def groundTruth(fixed_label_size, config):
    """
    create label with weight
    """
    neg_label = config.neg_label

    rPos = config.rPos / config.stride
    rNeg = config.rNeg / config.stride

    half = int(np.floor(fixed_label_size[0] / 2) + 1)

    if config.label_weight_method == "balanced":
        fixed_label = create_logisticloss_label(fixed_label_size, rPos, rNeg, neg_label)

        instance_weight = torch.ones(fixed_label.shape[0], fixed_label.shape[1])
        tmp_idx_P = np.where(fixed_label == 1)
        sumP = tmp_idx_P[0].size
        tmp_idx_N = np.where(fixed_label == neg_label)
        sumN = tmp_idx_N[0].size
        instance_weight[tmp_idx_P] = 0.5 * instance_weight[tmp_idx_P] / sumP
        instance_weight[tmp_idx_N] = 0.5 * instance_weight[tmp_idx_N] / sumN
        
        # reshape label
        fixed_label = fixed_label.clone().view(1, 1, fixed_label.shape[0], fixed_label.shape[1]).contiguous()
        # fixed_label = torch.reshape(fixed_label, (1, 1, fixed_label.shape[0], fixed_label.shape[1]))
        fixed_label = fixed_label.repeat(config.batchsize, 1, 1, 1)

        # reshape weight
        instance_weight = instance_weight.clone().view(1, instance_weight.shape[0], instance_weight.shape[1]).contiguous()
        # instance_weight = torch.reshape(instance_weight, (1, instance_weight.shape[0], instance_weight.shape[1]))

    if config.use_gpu:
        fixed_label, instance_weight = fixed_label.cuda(), instance_weight.cuda()
    
    return fixed_label, instance_weight


def cv2_brg2rgb(bgr_img):
    # convert brg image to rgb
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img


def float32_to_uint8(img):
    #convert float32 array to uint8
    beyong_255 = np.where(img > 255)
    img[beyong_255] = 255
    less_0 = np.where(img < 0)
    img[less_0] = 0
    img = np.round(img)
    return img.astype(np.uint8)