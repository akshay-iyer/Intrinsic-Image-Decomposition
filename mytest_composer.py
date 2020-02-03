import sys, os, argparse, torch, pdb, scipy.misc
import torchvision, torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

import models, pipeline



def vector_to_image(vector):
    # mask = make_mask(img)
    dim = vector.dim()
    ## batch
    if dim == 4:
        mask = torch.pow(vector,2).sum(1) > .01
        mask = mask.repeat(1,3,1,1)
    elif dim == 3:
        mask = torch.pow(vector,2).sum(0) > .01
        mask = mask.repeat(3,1,1)
    else:
        raise RuntimeError 
    img = vector.clone()
    img[mask] /= 2.
    img[mask] += .5
    # img = (vector / 2.) + 5.
    return img

# preparing necessary models
shader_path = "saved/shader/model.t7"
comp_path   = "saved/composer/rough/state.t7"
decomp_path = "saved/decomposer/state_fresh.t7"

decomposer = models.Decomposer().cuda()
shader = torch.load(shader_path).cuda()

model = models.Composer(decomposer, shader).cuda()
model.load_state_dict(torch.load(comp_path))
model.train(mode=False)


# reading and preprocessing images
inp  = scipy.misc.imread("akshay_tests_turing/wound_images/images/1.png")
refl = scipy.misc.imread("akshay_tests_turing/wound_images/images/3_albedo.png")
shad = scipy.misc.imread("akshay_tests_turing/wound_images/images/3_shading.png")
mask = scipy.misc.imread("akshay_tests_turing/wound_images/masks/1mask.png")

if refl.shape[-1] == 4:
    refl = refl[:,:,:-1]
refl = refl.transpose(2,0,1) / 255.

if shad.shape[-1] == 4:
    shad = shad[:,:,:-1]
shad = shad.transpose(2,0,1) / 255.

if mask.shape[-1] == 4:
    mask = mask[:,:,:-1]
mask = mask.transpose(2,0,1) / 255.

if inp.shape[-1] == 4:
    inp = inp[:,:,:-1]
inp = inp.transpose(2,0,1) / 255.


# feeding necessary inputs to model
inp  = Variable(torch.from_numpy(inp).float().cuda(async=True))
mask = Variable(torch.from_numpy(mask).float().cuda(async=True))

inp  = inp.unsqueeze(0).cuda()
mask = mask.unsqueeze(0).cuda()

print("size after adding extra dimension for batch size")
print(inp.size())
print(mask.size())

recon, refl_pred, depth_pred, shape_pred, lights_pred, shad_pred = model.forward(inp, mask)

criterion = nn.MSELoss(size_average=True).cuda()
recon_loss = criterion(recon, inp).data[0]
print("recon loss: ",recon_loss)

depth_pred = depth_pred.repeat(1,3,1,1)
shad_pred = shad_pred.repeat(1,3,1,1)
shape_pred = pipeline.vector_to_image(shape_pred)

images= []
splits = []
for tensor in [inp, recon, refl_pred, shape_pred, shad_pred]:
    splits.append( [img.squeeze() for img in tensor.data.split(1)] )
# pdb.set_trace()
splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
images.extend(splits)

for ind, i in enumerate(images):
    i = i.cpu().numpy().transpose(1,2,0)
    print(i.shape)
    fullpath = str(ind) + ".png"
    scipy.misc.imsave(fullpath, i)  

