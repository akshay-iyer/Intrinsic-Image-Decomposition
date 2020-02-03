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
parser.add_argument('--intrinsics',     type=list,  default=['input', 'mask', 'albedo', 'depth', 'normals', 'lights', 'shading'],
        help='intrinsic images to load from the test sets')
parser.add_argument('--save_path',      type=str,   default='saved/composer/inference/',
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


# preparing necessary models
# preparing necessary models
shader_path = "saved/shader/model.t7"
comp_path   = "saved/composer/composer_cl/state.t7"
decomp_path = "saved/decomposer/decomp_cl/state_cl.t7"

decomposer = models.Decomposer().cuda()
shader = torch.load(shader_path).cuda()

model = models.Composer(decomposer, shader).cuda()
model.load_state_dict(torch.load(comp_path))
model.train(mode=False)

test_set    = pipeline.IntrinsicDataset(args.data_path, args.test_sets, args.intrinsics, array=args.array, size_per_dataset=args.num_test, rel_path='../intrinsics-network')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.loaders, shuffle=False)

images = []

criterion = nn.MSELoss(size_average=True).cuda()
recon_loss = 0
refl_loss = 0
depth_loss = 0
shape_loss = 0
lights_loss = 0
shad_loss = 0
depth_normals_loss = 0

masks = []

for ind, tensors in enumerate(test_loader):
    tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
    inp, mask, refl_targ, depth_targ, shape_targ, lights_targ, shad_targ = tensors
    depth_normals_targ = pipeline.depth_to_normals(depth_targ.unsqueeze(1), mask=mask)
    # depth_normals_targ

    depth_targ = depth_targ.unsqueeze(1).repeat(1,3,1,1)
    shad_targ = shad_targ.unsqueeze(1).repeat(1,3,1,1)

    recon, refl_pred, depth_pred, shape_pred, lights_pred, shad_pred = model.forward(inp, mask)
    # relit = pipeline.relight(model.shader, shape_pred, lights_pred, 6)
    # relit_mean = relit.mean(0).squeeze()

    depth_normals_pred = pipeline.depth_to_normals(depth_pred, mask=mask)

    depth_pred = depth_pred.repeat(1,3,1,1)
    shad_pred = shad_pred.repeat(1,3,1,1)

    recon_loss += criterion(recon, inp).data[0]
    refl_loss += criterion(refl_pred, refl_targ).data[0]
    depth_loss += criterion(depth_pred, depth_targ).data[0]
    shape_loss += criterion(shape_pred, shape_targ).data[0]
    lights_loss += criterion(lights_pred, lights_targ).data[0]
    shad_loss += criterion(shad_pred, shad_targ).data[0]
    depth_normals_loss += lights_pred[:,1].sum() ##criterion(shape_pred, depth_normals_pred.detach()).data[0]

    # I commented out since can't run blender on turing
    lights_rendered_targ = render.vis_lights(lights_targ, verbose=False)
    lights_rendered_pred = render.vis_lights(lights_pred, verbose=False)
    # # pdb.set_trace()

    shape_targ = pipeline.vector_to_image(shape_targ)
    shape_pred = pipeline.vector_to_image(shape_pred)

    depth_normals_targ = pipeline.vector_to_image(depth_normals_targ)
    depth_normals_pred = pipeline.vector_to_image(depth_normals_pred)


    splits = []
    # pdb.set_trace()

    # removed lights_rendered_targ and lights_rendered_pred since can't run blender on turing
    for tensor in [ inp,    refl_targ,  depth_targ,     depth_normals_targ,     shape_targ,     shad_targ, lights_rendered_targ,  
                    recon,  refl_pred,  depth_pred,     depth_normals_pred,     shape_pred,     shad_pred, lights_rendered_pred ]:
                    # relit[0], relit[1], relit[2], relit[3], relit[4], relit[5], relit_mean]:
        splits.append( [img.squeeze() for img in tensor.data.split(1)] )

    masks.append(mask)

    # pdb.set_trace()
    # print shad_targ.size()
    # print shad_pred.size()
    # print [len(sublist) for sublist in splits]
    splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
    images.extend(splits)

labels = [  'recon_targ', 'refl_targ', 'depth_targ', 'depth_normals_targ', 'shape_targ', 'shad_targ', 'lights_targ',
            'recon_pred', 'refl_pred', 'depth_pred', 'depth_normals_pred', 'shape_pred', 'shad_pred', 'lights_pred']
          
masks = [i.split(1) for i in masks]
masks = [item.squeeze()[0].unsqueeze(0).data.cpu().numpy().transpose(1,2,0) for sublist in masks for item in sublist]



grid_path = os.path.join(args.save_path, 'trained.png')



# it is called after loss for all imgs in all val batches has been added. thus ind+1 would be equal to the number of batches  
recon_loss /= float(ind+1)
refl_loss /= float(ind+1)
depth_loss /= float(ind+1)
shape_loss /= float(ind+1)
lights_loss /= float(ind+1)
shad_loss /= float(ind+1)
depth_normals_loss /= float(ind+1)
depth_normals_loss = depth_normals_loss.data[0]

losses = [recon_loss, refl_loss, depth_loss, shape_loss, lights_loss, shad_loss]
# pdb.set_trace()
grid = torchvision.utils.make_grid(images, nrow=7).cpu().numpy().transpose(1,2,0)
#grid = np.clip(grid, 0, 1)
# fullpath = os.path.join(save_path, str(epoch) + '.png')
scipy.misc.imsave(grid_path, grid)
print('<Test> Losses: ', losses)

