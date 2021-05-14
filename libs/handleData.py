import numpy as np, math
import cv2, json, random
from os import path
from tensorflow.keras.utils import Sequence
import torch
from torchvision import transforms
from libs.utils import buildResponseMap

class DataGenerator(Sequence):
	def __init__(self, datapath, metadatapath, batchsize):
		self.batchsz = batchsize
		self.samplerate = 10
		self.pairintv = 10

		self.ishape = (265,265)
		self.xshape = (255,255)
		self.zshape = (127,127)
		self.shiftr = (self.ishape[0]-self.xshape[0], self.ishape[1]-self.xshape[1])
		self.stride = 1
		self.rshape = ((self.xshape[0]-self.zshape[0])//self.stride+1, (self.xshape[1]-self.zshape[1])//self.stride+1)

		self.datapath = datapath
		self.metadata = json.load(open(metadatapath, 'r'))
		self.videoNames = list(self.metadata.keys())
		
		self.pickPairs()

	def __len__(self):
		nbatches = math.ceil(len(self.pairs)/self.batchsz)
		return nbatches

	def __getitem__(self, idx):
		batchData = self.pairs[idx*self.batchsz : (idx+1)*self.batchsz]
		z_images = [cv2.imread(imagepath) for imagepath in batchData[:, 0]]
		x_images = [cv2.imread(imagepath) for imagepath in batchData[:, 1]]

		xshift = [(random.randrange(-self.shiftr[0]//2,self.shiftr[0]//2+1), random.randrange(-self.shiftr[1]//2,self.shiftr[1]//2+1)) 
				  for n in range(len(x_images))]

		zbbox = (self.ishape[1]//2-self.zshape[1]//2, self.ishape[0]//2-self.zshape[0]//2, 
				 self.ishape[1]//2+self.zshape[1]//2, self.ishape[0]//2+self.zshape[0]//2)
		xbbox = (self.ishape[1]//2-self.xshape[1]//2, self.ishape[0]//2-self.xshape[0]//2, 
				 self.ishape[1]//2+self.xshape[1]//2, self.ishape[0]//2+self.xshape[0]//2)

		response = buildResponseMap(self.zshape, self.ishape, self.stride)
		z_images = [img[zbbox[1]:zbbox[3]+1, zbbox[0]:zbbox[2]+1, :] 
							  for img in z_images]
		x_images = [img[xbbox[1]+y:xbbox[3]+y+1, xbbox[0]+x:xbbox[2]+x+1, :] 
							  for img, (y,x) in zip(x_images, xshift)]

		rbbox = (response.shape[1]//2, response.shape[0]//2)
		r_images = [response[rbbox[1]+y-self.rshape[1]//2:rbbox[1]+y+self.rshape[1]//2+1, 
							 rbbox[0]+x-self.rshape[0]//2:rbbox[0]+x+self.rshape[0]//2+1] 
					for y,x in xshift]

		return self.imgTransforms(x_images), self.imgTransforms(z_images), torch.from_numpy(r_images)

	def imgTransforms(self, images):
		composer = transforms.Compose([ transforms.ToTensor() ])
		imageTensors = [composer(image).unsqueeze(0) for image in images]
		imageTensors = torch.cat(imageTensors, dim=0)
		return imageTensors

	def pickPairs(self):
		self.pairs = []
		for videoName in self.videoNames:
			nframes = self.metadata[videoName].get('nframes')
			frames = self.metadata[videoName].get('frames')

			z_images = random.sample(range(nframes), self.samplerate)
			x_images = [random.randrange(max(0, n-self.pairintv), min(nframes, n+self.pairintv)) for n in z_images]

			z_images = [path.join(self.datapath, frames[i]) for i in z_images]
			x_images = [path.join(self.datapath, frames[i]) for i in x_images]

			self.pairs+= list(map(lambda z,x:[z,x], z_images, x_images))
		
		random.shuffle(self.pairs)
		self.pairs = np.asarray(self.pairs)

	def on_epoch_end(self):
		self.pickPairs()