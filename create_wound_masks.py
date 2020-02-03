import sys, os, scipy.misc
import matplotlib.pyplot as plt
import numpy as np
path = "/home/abiyer/intrinsics-network/dataset/output/GroundTruth"

for file in os.listdir(path):
	image = scipy.misc.imread(os.path.join(path,file))
	image = scipy.misc.imresize(image,(256,256))
	wmask = (image == [255.,0.,0.]).all(axis=2)
	smask = (image == [255.,255.,0.]).all(axis=2)
	omask = (image == [0.,255.,0.]).all(axis=2)
	# remaining mask, for rest of the pixels
	rmask = np.logical_and(image != [0.,0.,0.], image != [255.,255.,255.]).all(axis=2)
	#rmask = (image > [0.,0.,0.]).all(axis=2)
	

	image[wmask] = [255,255,255]
	image[smask] = [255,255,255]
	#image[rmask] = [255,255,255]
	image[omask] = [0,0,0]
	# plt.imshow(np.uint8(rmask))
	# plt.show()
	print(file)
	scipy.misc.imsave(os.path.join(path,file),image)

