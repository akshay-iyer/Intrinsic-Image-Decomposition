import os
import scipy.misc
from PIL import Image
from skimage.transform import resize

path = "/home/abiyer/intrinsics-network/dataset/output/wound_train"
for file in os.listdir(path):
	img = scipy.misc.imread(os.path.join(path,file))
	#img = Image.open(os.path.join(path,file))
	
	#print ("shape before resize: ", img.shape)
	
	img = img.transpose(2,0,1) / 255.
	#scipy.misc.imresize(image,(256,256))
	#img.resize((256,256))
	#img = resize(img, (256, 256), anti_aliasing=True)
	print ("shape after resize: ", img.shape)

	scipy.misc.imsave(os.path.join(path,file)	, img)

	# token = file.split('.')
	# #new_filename = file.replace('_composite.png','_composite')
	# new_filename = token[0] + '_mask.' + token[1]
	# os.rename(os.path.join(path,file),os.path.join(path,new_filename))
	