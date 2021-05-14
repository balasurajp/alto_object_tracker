import numpy as np
import cv2


def mxmywh2xyxy(bbox):
	xmin = bbox[0]
	ymin = bbox[1]
	xmax = bbox[0]+bbox[2]
	ymax = bbox[1]+bbox[3]
	return (xmin, ymin, xmax, ymax)

def mxmywh2cxcywh(bbox):
	if bbox[2]%2==0:
		bbox[2]=bbox[2]+1
	if bbox[3]%2==0:
		bbox[3]=bbox[3]+1

	cx = bbox[0] + bbox[2]//2
	cy = bbox[1] + bbox[3]//2
	w = bbox[2]
	h = bbox[3]
	return (cx, cy, w, h)


def cropWithPadding(img, bbox, modelsz):
    imH, imW, _ = img.shape
    xmin, ymin, xmax, ymax = bbox

    left = right = top = bottom = 0
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > imW:
        right = int(xmax - imW)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > imH:
        bottom = int(ymax - imH)

    xmin = int(max(0, xmin))
    xmax = int(min(imW, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(imH, ymax))
    imgpatch = img[ymin:ymax+1, xmin:xmax+1]
    
    if left != 0 or right !=0 or top!=0 or bottom!=0:
        imgmean = tuple(map(int, img.mean(axis=(0, 1))))
        imgpatch = cv2.copyMakeBorder(imgpatch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=imgmean)
    
    impatch = cv2.resize(imgpatch, (modelsz, modelsz), interpolation = cv2.INTER_CUBIC)
    return impatch


def getSearchAndTarget(img, bbox, sizeT=127, sizeS=255):
	cx, cy, w, h = mxmywh2cxcywh(bbox)

	scalew = sizeT/w
	scaleh = sizeT/h

	wS = round(sizeS/scalew)
	hS = round(sizeS/scaleh)

	xminS = cx - wS//2
	yminS = cy - hS//2
	xmaxS = cx + wS//2
	ymaxS = cy + hS//2

	impatch = cropWithPadding(img, bbox=(xminS, yminS, xmaxS, ymaxS), modelsz=sizeS)
	return impatch, (scalew, scaleh)