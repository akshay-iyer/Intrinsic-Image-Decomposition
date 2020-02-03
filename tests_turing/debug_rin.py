import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image


import sys, os, argparse, pdb
sys.path.append('..')
import  pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default='dataset/output/',
        help='base path for datasets')
parser.add_argument('--train_sets',     type=str,   default='motorbike_train,bottle_train',
        help='folders within data_path to draw from during training')
parser.add_argument('--val_sets',       type=str,   default='motorbike_val,bottle_val',
        help='folders within data_path to draw from during validation')
parser.add_argument('--intrinsics',     type=list,  default=['input', 'mask', 'albedo', 'depth', 'normals', 'lights'],
        help='intrinsic images to load from the train and val sets')
parser.add_argument('--save_path',      type=str,   default='components/test_logger/',
        help='save folder for model, plots, and visualizations')
parser.add_argument('--lr',             type=float, default=0.01,
        help='learning rate')
parser.add_argument('--num_epochs', type=int,   default=500,
        help='number of training epochs')
parser.add_argument('--lights_mult',    type=float, default=0.01,
        help='multiplier on the lights loss')
parser.add_argument('--array',          type=str,   default='shader',
        help='array with lighting parameters')
parser.add_argument('--num_train',  type=int,   default=100,
        help='number of training images per object category')
parser.add_argument('--num_val',    type=int,   default=100,
        help='number of validation images per object category')
parser.add_argument('--loaders',    type=int,   default=4,
        help='number of parallel data loading processes')
parser.add_argument('--batch_size',    type=int,   default=128)
args = parser.parse_args()



train_set = pipeline.IntrinsicDataset(args.data_path, args.train_sets, args.intrinsics, array=args.array, size_per_dataset=args.num_train, rel_path= '../')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loaders, shuffle=False)

val_set = pipeline.IntrinsicDataset(args.data_path, args.val_sets, args.intrinsics, array=args.array, size_per_dataset=args.num_val, rel_path='../')
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.loaders, shuffle=False)

for ind,(images, labels) in train_loader:
    grid = torchvision.utils.make_grid(images, nrow=32)
    torchvision.utils.save_image(grid, os.path.join('loader_imgs', 'trgbatch' + ind +'.png'))

# print ("test")
# ''' in the foll snippet, I'm just checking MSE loss value for same predicted and target image to check if it gives 0'''
# criteria = nn.MSELoss(size_average=True).cuda()
# inputt = Image.open("/home/abiyer/intrinsics-network/dataset/output/car_val/1_albedo.png")
# output = Image.open("/home/abiyer/intrinsics-network/dataset/output/car_val/2_albedo.png")
#
# trans  = torchvision.transforms.ToTensor()
# inputt = Variable(trans(inputt))
# output = Variable(trans(output))
#
# loss = criteria(inputt,output)
# print("loss: ", loss.data)