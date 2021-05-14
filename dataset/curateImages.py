import initpaths

import numpy as np, cv2, json
from glob import glob
from os import path, makedirs, listdir
from pathlib import Path
from libs.imageOps import getSearchAndTarget

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


def determineZeroBboxVideos(videospath, videoNames):
	book = {'video':[], 'frames':[]}
	for videoName in videoNames:
		collectframes = []
		bboxes = getBoundingBoxes(path.join(videospath, videoName, 'groundtruth.txt'))
		for i, bbox in enumerate(bboxes):
			width = bbox[2]
			height = bbox[3]
			if width==0 or height==0:
				collectframes.append(i)
		if len(collectframes)!=0:
			book['video'].append(videoName)
			book['frames'].append(collectframes)
	return book

def getBoundingBoxes(filepath):
	file = open(filepath, 'r')
	lines = file.readlines()
	lines = [line.strip('\n').split(',') for line in lines]
	bboxs = [[int(float(x)) for x in line] for line in lines]
	return bboxs

def processVideo(srcpath, dstpath, searchsz, targetsz, videoName):
	makedirs(path.join(dstpath, videoName), exist_ok = True)

	frameNames = sorted(glob(path.join(srcpath, videoName, '*.jpg')))
	frameBboxs = getBoundingBoxes(path.join(srcpath, videoName, 'groundtruth.txt'))

	metadata = {'frames':[], 'scales':[], 'nframes':None}
	for bbox, framepath in zip(frameBboxs, frameNames):
		imgframe  = cv2.imread(framepath)
		frameName = path.split(framepath)[-1] 

		# print(videoName, '-->', frameName)
		imgpatch, scales = getSearchAndTarget(imgframe, bbox, sizeT=targetsz, sizeS=searchsz)
		
		cv2.imwrite(path.join(dstpath, videoName, frameName), imgpatch)
		metadata['frames'].append(path.join(videoName, frameName))
		metadata['scales'].append(scales)

	metadata['nframes'] = len(frameNames) 
	return videoName, metadata

if __name__== "__main__":
	numThreads = 24

	trndpath = '/media/hd1/suraj/datasets/got10k/train'
	trnOpath = '/media/hd1/suraj/datasets/processed/got/train'
	trnvideoNames = sorted(listdir(trndpath))[:-1]

	book = determineZeroBboxVideos(trndpath, trnvideoNames)
	print(book)
	makedirs(trnOpath, exist_ok = True)
	
	valdpath = '/media/hd1/suraj/datasets/got10k/val'
	valOpath = '/media/hd1/suraj/datasets/processed/got/val'
	valvideoNames = sorted(listdir(valdpath))[:-1]
	
	book = determineZeroBboxVideos(valdpath, valvideoNames)
	print(book)
	makedirs(valOpath, exist_ok = True)

	zsize = 127
	xsize = 255+10

	#--------------------------------------------------------------------------------------------------------------------------------------------
	print('Processing trainingVideos started...............')
	partialProcessVideo = partial(processVideo, trndpath, trnOpath, xsize, zsize)

	jsonfile = dict()
	with Pool(processes=numThreads) as pool:
		poolprocess = pool.imap_unordered(partialProcessVideo, trnvideoNames)
		for vName, metadata in tqdm(poolprocess, total=len(trnvideoNames)):
			jsonfile[vName] = metadata
	json.dump(jsonfile, open('trnMetadata.json', 'w'))

	#--------------------------------------------------------------------------------------------------------------------------------------------
	print('Processing validationVideos started...............')
	partialProcessVideo = partial(processVideo, valdpath, valOpath, xsize, zsize)
	
	jsonfile = dict()
	with Pool(processes=numThreads) as pool:
		poolprocess = pool.imap_unordered(partialProcessVideo, valvideoNames)
		for vName, metadata in tqdm(poolprocess, total=len(valvideoNames)):
			jsonfile[vName] = metadata
	json.dump(jsonfile, open('valMetadata.json', 'w'))
