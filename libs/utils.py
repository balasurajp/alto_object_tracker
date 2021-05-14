import numpy as np
from libs.evaluationOps import bbIOU

def buildResponseMap(zshape, xshape, stride):
	rshape = ((xshape[0]-zshape[0])//stride+1, (xshape[1]-zshape[1])//stride+1)
	response = np.zeros(rshape)
	
	gtbbox = (xshape[1]//2-127//2, xshape[0]//2-127//2, xshape[1]//2+127//2, xshape[0]//2+127//2)
	for y in range(rshape[1]):
		for x in range(rshape[0]):
			pdbbox = (0+x, 0+y, zshape[1]+x-1, zshape[0]+y-1)
			response[x,y] = bbIOU(pdbbox, gtbbox)
	return response