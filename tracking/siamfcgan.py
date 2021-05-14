from __future__ import absolute_import, division

from experiments.siamese_fc.Config import Config
from experiments.siamese_fc.network import SiamNet_G

import cv2
import epdb
import datetime
import numpy as np
from easydict import EasyDict as edict

import torch
from torch.autograd import Variable
from lib.Tracking_Utils import *

from got10k.trackers import Tracker

class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__(
            name='ALTO', is_deterministic=True)
        self.cfg = Config()

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:3' if self.cuda else 'cpu')

        # setup model
        self.net = SiamNet_G()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)


    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_UP * self.cfg.score_size
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.num_scale // 2),
            self.cfg.num_scale // 2, self.cfg.num_scale)

        # exemplar and search sizes
        context = self.cfg.context_amount * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_size / self.cfg.examplar_size

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.examplar_size,
            pad_color=self.avg_color)

        # exemplar features
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel = self.net.feat_extraction(exemplar_image)
            self.kernel = self.kernel.repeat(self.cfg.num_scale, 1, 1, 1)

    def update(self, image):
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_size,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            instances = self.net.feat_extraction(instance_images)
            responses = self.net.xcorr(self.kernel, instances)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.num_scale // 2] *= self.cfg.scale_penalty
        responses[self.cfg.num_scale // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.w_influence) * response + \
            self.cfg.w_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.stride / self.cfg.response_UP
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_size
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_LR) * 1.0 + \
            self.cfg.scale_LR * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch