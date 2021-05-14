"""
Dataset for VID
Written by Heng Fan
"""

from torch.utils.data.dataset import Dataset
import json
from .Utils import viewimages,extract_Dpatches, cv2_brg2rgb,float32_to_uint8
import os
import numpy as np
import cv2


class VIDDataset(Dataset):

    def __init__(self, config, transforms, mode="Train"):
        if(mode == 'Train'):
            imdb_video = json.load(open(config.trnimdb, 'r'))
        else:
            imdb_video = json.load(open(config.valimdb, 'r'))

        self.videos     = imdb_video['videos']
        self.data_dir   = config.datapath
        self.config     = config
        self.num_videos = int(imdb_video['num_videos'])
        self.center_range = (127,254) # end value not included

        self.z_transforms = transforms[0]
        self.x_transforms = transforms[1]     

    def __getitem__(self, rand_vid):
        '''
        read a pair of images z and x
        '''
        # randomly decide the id of video to get z and x
        rand_vid = rand_vid % self.num_videos

        video_keys = list(self.videos.keys())
        video = self.videos[video_keys[rand_vid]]

        # get ids of this video
        video_ids = video[0]
        # how many ids in this video
        video_id_keys = list(video_ids.keys())

        # randomly pick an id for z
        rand_trackid_z = np.random.choice(list(range(len(video_id_keys))))
        # get the video for this id
        video_id_z = video_ids[video_id_keys[rand_trackid_z]]

        # pick a valid examplar z in the video
        rand_z = np.random.choice(range(len(video_id_z)))

        # pick a valid instance within frame_range frames from the examplar, excluding the examplar itself
        possible_x_pos = list(range(len(video_id_z)))
        rand_x = np.random.choice(possible_x_pos[max(rand_z - self.config.pairband, 0):rand_z] + possible_x_pos[(rand_z + 1):min(rand_z + self.config.pairband, len(video_id_z))])

        z = video_id_z[rand_z].copy()    # use copy() here to avoid changing dictionary
        x = video_id_z[rand_x].copy()

        # read z and x
        img_z = cv2.imread(os.path.join(self.data_dir, z['instance_path']))
        img_z = self.__cvt_color(img_z)

        img_x = cv2.imread(os.path.join(self.data_dir, x['instance_path']))
        img_x = self.__cvt_color(img_x)

        # Data augmentation:
        # NOTE: We have done center crop for z in the data augmentation
        img_z = self.z_transforms(img_z)
        img_x = self.x_transforms(img_x)

        return img_z, img_x

    def __len__(self):
        return int(self.config.numpairs)

    def __cvt_color(self, img):
        if img.shape[-1] == 1:
            img = np.tile(img, (1, 1, 3))
        else:
            # BGR to RGB
            img = img[:, :, ::-1]
        return img