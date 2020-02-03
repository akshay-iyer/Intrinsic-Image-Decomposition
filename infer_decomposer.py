import sys, os, argparse, torch, pdb, scipy.misc
import torchvision, torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

import math, torch, torchvision.utils, numpy as np
from tqdm import tqdm

import models, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default='/dataset/output',
        help='base path for datasets')
parser.add_argument('--test_sets',     type=str,   default='airplane_test',
        help='folders within data_path to draw from during test')
parser.add_argument('--intrinsics',     type=list,  default=['input', 'mask', 'albedo', 'depth', 'normals', 'lights'],
        help='intrinsic images to load from the test sets')
parser.add_argument('--save_path',      type=str,   default='saved/decomposer/inference/',
        help='save folder for model, plots, and visualizations')
parser.add_argument('--lr',             type=float, default=0.01,
        help='learning rate')
parser.add_argument('--array',          type=str,   default='shader',
        help='array with lighting parameters')
parser.add_argument('--num_test',  type=int,   default=100,
        help='number of test images per object category')
parser.add_argument('--loaders',    type=int,   default=4,
        help='number of parallel data loading processes')
parser.add_argument('--batch_size',    type=int,   default=32)
args = parser.parse_args()

render = pipeline.Render()

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

# losses 
criterion = nn.MSELoss(size_average=True).cuda()
refl_loss = 0
shape_loss = 0
lights_loss = 0

# preparing necessary models
#decomp_path = "saved/decomposer/decomp_with_dropout/state_dropout.t7"
decomp_path = "saved/decomposer/decomp_cl/state_cl.t7"

decomposer = models.Decomposer().cuda()
checkpoint = torch.load(decomp_path)
#decomposer.load_state_dict(checkpoint['model_state_dict'])
decomposer.load_state_dict(checkpoint)
decomposer.train(mode=False)

test_set    = pipeline.IntrinsicDataset(args.data_path, args.test_sets, args.intrinsics, array=args.array, size_per_dataset=args.num_test, rel_path='../intrinsics-network')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.loaders, shuffle=False)

images = []

for ind, tensors in enumerate(test_loader):
    tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
    inp, mask, refl_targ, depth_targ, shape_targ, lights_targ = tensors

    refl_pred, depth_pred, shape_pred, lights_pred = decomposer.forward(inp, mask)

    refl_loss += criterion(refl_pred, refl_targ).data[0]
    shape_loss += criterion(shape_pred, shape_targ).data[0]
    lights_loss += criterion(lights_pred, lights_targ).data[0]

    shape_targ = pipeline.vector_to_image(shape_targ)
    shape_pred = pipeline.vector_to_image(shape_pred)

    depth_targ = depth_targ.unsqueeze(1).repeat(1,3,1,1)
    depth_pred = depth_pred.repeat(1,3,1,1)

    lights_rendered_targ = render.vis_lights(lights_targ, verbose=False)
    lights_rendered_pred = render.vis_lights(lights_pred, verbose=False)
    # pdb.set_trace()
    splits = []
    for tensor in [inp, refl_pred, refl_targ, depth_pred, depth_targ, shape_pred, shape_targ, lights_rendered_pred, lights_rendered_targ]:
        splits.append( [img.squeeze() for img in tensor.data.split(1)] )
    # pdb.set_trace()
    splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
    images.extend(splits)


refl_loss /= float(ind+1)
shape_loss /= float(ind+1)
lights_loss /= float(ind+1)

# pdb.set_trace()
grid = torchvision.utils.make_grid(images, nrow=9).cpu().numpy().transpose(1,2,0)
grid = np.clip(grid, 0, 1)


fullpath = os.path.join(args.save_path, 'prediction.png')

scipy.misc.imsave(fullpath, grid)

losses = [refl_loss, shape_loss, lights_loss]
print('<Test> Losses: ', losses)


# # reading and preprocessing images
# #inp  = scipy.misc.imread("akshay_tests_turing/wound_images/images/1_composite.png")
# refl = scipy.misc.imread("akshay_tests_turing/wound_images/images/1_albedo.png")
# shad = scipy.misc.imread("akshay_tests_turing/wound_images/images/1_shading.png")
# mask = scipy.misc.imread("akshay_tests_turing/wound_images/masks/1_mask.png")



# if refl.shape[-1] == 4:
#     refl = refl[:,:,:-1]
# refl = refl.transpose(2,0,1) / 255.

# if shad.shape[-1] == 4:
#     shad = shad[:,:,:-1]
# shad = shad.transpose(2,0,1) / 255.

# if mask.shape[-1] == 4:
#     mask = mask[:,:,:-1]
# mask = mask.transpose(2,0,1) / 255.

# inp = refl*shad

# # if inp.shape[-1] == 4:
# #     inp = inp[:,:,:-1]
# # inp = inp.transpose(2,0,1) / 255.


# # feeding necessary inputs to model
# inp  = Variable(torch.from_numpy(inp).float().cuda(async=True))
# mask = Variable(torch.from_numpy(mask).float().cuda(async=True))

# inp  = inp.unsqueeze(0).cuda()
# mask = mask.unsqueeze(0).cuda()

# print("size after adding extra dimension for batch size")
# print(inp.size())
# print(mask.size())

# refl_pred, depth_pred, shape_pred, lights_pred = decomposer.forward(inp, mask)
# shape_pred = pipeline.vector_to_image(shape_pred)


# images= []
# splits = []
# for tensor in [inp, refl_pred, shape_pred]:
#     splits.append( [img.squeeze() for img in tensor.data.split(1)] )
# # pdb.set_trace()
# splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
# images.extend(splits)

# for ind, i in enumerate(images):
#     i = i.cpu().numpy().transpose(1,2,0)
#     print(i.shape)
#     fullpath = str(ind) + ".png"
#     scipy.misc.imsave(fullpath, i)  