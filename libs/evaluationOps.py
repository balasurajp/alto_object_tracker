import numpy as np

def bbIOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# intersection area
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# boxA and boxB areas
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the IOU
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

