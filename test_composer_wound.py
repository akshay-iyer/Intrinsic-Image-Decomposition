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

shader_path = "saved/shader/model.t7"
comp_path   = "saved/composer/rough/state.t7"
decomp_path = "saved/decomposer/state_fresh.t7"
decomposer = models.Decomposer().cuda()

shader = torch.load(shader_path).cuda()

# for composer
# model = models.Composer(decomposer, shader).cuda()
# model.load_state_dict(torch.load(comp_path))
# model.eval()

# for decomposer
decomposer.load_state_dict(torch.load(decomp_path))
decomposer.train(mode=False)

#print(model.state_dict)
# inp  = Image.open("akshay_tests_turing/wound_images/images/20_composite.png")
# mask = Image.open("akshay_tests_turing/wound_images/masks/3_mask.png")

# refl = Image.open("akshay_tests_turing/wound_images/images/3_albedo.png")
# shad = Image.open("akshay_tests_turing/wound_images/images/3_shading.png")
# inp  = ImageChops.multiply(refl,shad)

inp  = scipy.misc.imread("akshay_tests_turing/wound_images/images/3_albedo.png")
refl = scipy.misc.imread("akshay_tests_turing/wound_images/images/3_albedo.png")
shad = scipy.misc.imread("akshay_tests_turing/wound_images/images/3_shading.png")
mask = scipy.misc.imread("akshay_tests_turing/wound_images/masks/3_mask.png")

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

#inp = refl*shad

# inp  = inp.resize((256,256))
# mask = mask.resize((256,256))

# print("size of just image")
# print(inp.size)
# print(mask.size)

# print("channels")
# print(inp.getbands())
# print(mask.getbands())

pil_to_torch = transforms.ToTensor()
torch_to_pil = transforms.ToPILImage()

# inp  = pil_to_torch(inp)
# mask = pil_to_torch(mask)

# print("size after converting to tensor")
# print(inp.size())
# print(mask.size())

# inp  = inp[:-1,:,:]
# mask = mask[:-1,:,:]

# print("channels after removing alpha")
# print(inp.size())
# print(mask.size())

inp  = Variable(torch.from_numpy(inp).float().cuda(async=True))
mask = Variable(torch.from_numpy(mask).float().cuda(async=True))

# inp  = torch.squeeze(Variable(inp))
# mask = torch.squeeze(Variable(mask))

inp  = inp.unsqueeze(0).cuda()
mask = mask.unsqueeze(0).cuda()

print("size after adding extra dimension for batch size")
print(inp.size())
print(mask.size())

# inp  = inp.view(1,3,256,256)
# mask = mask.view(1,3,256,256)



# for composer
# recon, refl_pred, depth_pred, shape_pred, lights_pred, shad_pred = model.forward(inp, mask)

# recon = torch.squeeze(recon)
# recon = torch_to_pil(recon.data.cpu())
# print("recon size")
# print(recon.size)
# recon.save("recon.png")

# refl_pred = torch.squeeze(refl_pred)
# refl_pred = torch_to_pil(refl_pred.data.cpu())
# refl_pred.save("refl_pred.png")

# shape_pred = torch.squeeze(shape_pred)
# shape_pred = torch_to_pil(shape_pred.data.cpu())
# shape_pred.save("shape_pred.png")

# depth_pred = torch.squeeze(depth_pred)
# depth_pred = depth_pred.repeat(3,1,1)
# print("depth_pred size")
# print(depth_pred.size())
# depth_pred = torch_to_pil(depth_pred.data.cpu())
# depth_pred.save("depth_pred.png")

# shad_pred = torch.squeeze(shad_pred)

# shad_pred = shad_pred.repeat(3,1,1)
# print("shad_pred size")
# print(shad_pred.size())
# shad_pred = torch_to_pil(shad_pred.data.cpu())
# shad_pred.save("shad_pred.png")

# for decomposer
refl_pred, depth_pred, shape_pred, lights_pred = decomposer.forward(inp, mask)
shape_pred = pipeline.vector_to_image(shape_pred)


images= []
splits = []
for tensor in [inp, refl_pred, shape_pred]:
    splits.append( [img.squeeze() for img in tensor.data.split(1)] )
# pdb.set_trace()
splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
images.extend(splits)

for ind, i in enumerate(images):
	i = i.cpu().numpy().transpose(1,2,0)
	print(i.shape)
	fullpath = str(ind) + ".png"
	scipy.misc.imsave(fullpath, i)	


# print(type(images[0]))
# grid = torchvision.utils.make_grid(images, nrow=4).cpu().numpy().transpose(1,2,0)
# grid = np.clip(grid, 0, 1)

# fullpath = "grid.png"
# scipy.misc.imsave(fullpath, grid)

# refl_pred = torch.squeeze(refl_pred)
# refl_pred = torch_to_pil(refl_pred.data.cpu())
# refl_pred.save("refl_pred.png")


# shape_pred = torch.squeeze(shape_pred)
# shape_pred = torch_to_pil(shape_pred.data.cpu())
# shape_pred.save("shape_pred.png")

# depth_pred = torch.squeeze(depth_pred)
# depth_pred = depth_pred.repeat(3,1,1)
# print("depth_pred size")
# print(depth_pred.size())
# depth_pred = torch_to_pil(depth_pred.data.cpu())
# depth_pred.save("depth_pred.png")