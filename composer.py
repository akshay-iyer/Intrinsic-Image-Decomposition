
import sys, os, argparse, torch, pdb

import models, pipeline

from pytorchtools import EarlyStopping


parser = argparse.ArgumentParser()
parser.add_argument('--decomposer',         type=str,   default='saved/vector_depth_state_decomp_0.001lr_0.1lights/state.t7',
        help='decomposer network state file')
parser.add_argument('--shader',             type=str,   default='saved/vector_shader_0.01/model.t7',
        help='shader network file')
parser.add_argument('--data_path',          type=str,   default='../dataset/output/',
        help='base folder of datasets')
parser.add_argument('--unlabeled',          type=str,   default='car_train',
        help='unlabeled dataset(s), separated by commas, to use during training')
parser.add_argument('--labeled',            type=str,   default='motorbike_train,airplane_train,bottle_train',
        help='labeled dataset(s), separated by commas, to use during training')
parser.add_argument('--val_sets',           type=str,   default='car_val,motorbike_val',
        help='validation dataset(s), separated by commas')
parser.add_argument('--val_intrinsics',     type=list,  default=['input', 'mask', 'albedo', 'depth', 'normals', 'lights', 'shading'],
        help='intrinsic images to load for validation sets')
parser.add_argument('--save_path',          type=str,   default='logs/composer/',
        help='save path of model, visualizations, and error plots')
parser.add_argument('--labeled_array',      type=str,   default='shader',
        help='array of lighting parameters for unlabeled data')
parser.add_argument('--unlabeled_array',    type=str,   default='shader',
        help='array of lighting parameters for labeled data')
parser.add_argument('--lr',                 type=float, default=0.01,
        help='learning rate')
parser.add_argument('--num_val',            type=int,   default=10,
        help='number of validation images')
parser.add_argument('--lights_mult',        type=float, default=1,
        help='multiplier on lights loss')
parser.add_argument('--un_mult',            type=float, default=1,
        help='multipler on reconstruction loss')
parser.add_argument('--lab_mult',           type=float, default=1,
        help='multipler on labeled intrinsic images loss')
parser.add_argument('--loader_threads',     type=float, default=4,
        help='number of parallel data-loading threads')
parser.add_argument('--save_model',         type=bool,  default=True,
        help='whether to save model or not')
parser.add_argument('--transfer',           type=str,   default='10-normals,shader_10-shader',
        help='specifies which parameters are updated and for how many epochs')
parser.add_argument('--iters',              type=int,   default=1,
        help='number of expected times an image is reused during an epoch')
parser.add_argument('--set_size',           type=int,   default=10000,
        help='number of images per training dataset')
parser.add_argument('--val_offset',         type=int,   default=5,
        help='number of images per validation set which are used in visualizations')
parser.add_argument('--epoch_size',         type=int,   default=3200,
        help='number of images in an epoch')
parser.add_argument('--num_epochs',         type=int,   default=300)
parser.add_argument('--batch_size',         type=float, default=32)
# added by me
parser.add_argument('--composer',         type=str,   default='saved/composer/composer_with_dropout/state_dropout.t7',
        help='composer network state file with dropout added by me')
args = parser.parse_args()

pipeline.initialize(args)

## decomposer : image --> reflectance, normals, lighting
decomposer = models.Decomposer().cuda()

# should be there if we are trg composer from scratch. removed since only training alb of already trained composer
# checkpoint = torch.load(args.decomposer)
# decomposer.load_state_dict(checkpoint['model_state_dict'])

# now loading old decomposer saved w/o checkpoint
decomposer.load_state_dict(torch.load(args.decomposer))

## shader : normals, lighting --> shading
# should be there if we are trg composer from scratch. 
shader = torch.load(args.shader)

# this has to be removed when trg composer from scratch
#shader = models.Shader().cuda()

## composer : image --> reflectance, normals, lighting, shading --> image
model = models.Composer(decomposer, shader).cuda()

# these 2 lines should be there only if we are resuming trg on composer
# checkpoint = torch.load(args.composer)
# model.load_state_dict(checkpoint['model_state_dict'])
# print("loaded previously saved composer")

## data loader for train set, which includes half labeled and half unlabeled data
train_set = pipeline.ComposerDataset(args.data_path, args.unlabeled, args.labeled, unlabeled_array=args.unlabeled_array, labeled_array=args.labeled_array, size_per_dataset=args.set_size)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(args.batch_size / 2), num_workers=args.loader_threads, shuffle=True)

## data loader for val set, which is completely labeled
val_set = pipeline.IntrinsicDataset(args.data_path, args.val_sets, args.val_intrinsics, inds=list(range(0,args.num_val*args.val_offset,args.val_offset)), array=args.unlabeled_array, size_per_dataset=50)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)

# wound_val_set    = pipeline.IntrinsicDataset(args.data_path, 'wound_val' ,['input', 'mask'], inds=list(range(0,args.num_val*args.val_offset,args.val_offset)), array=args.unlabeled_array, size_per_dataset=int(args.set_size/100), iswound = True)
# wound_val_loader = torch.utils.data.DataLoader(wound_val_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)

## print out error plots after every epoch for every prediction
logger = pipeline.Logger(['recon', 'refl', 'depth', 'shape', 'lights', 'shading'], args.save_path)
param_updater = pipeline.ParamUpdater(args.transfer)

# refl_es  = EarlyStopping()
# shape_es = EarlyStopping()
# light_es = EarlyStopping()
# shad_es  = EarlyStopping()

sub_epoch = 0
stop = False
for epoch in range(args.num_epochs):
	# checking if all nws overfitting. if yes then stop full training
	if stop:
		break

	print('<Main> Epoch {}'.format(epoch))
	sub_epoch = 0
	
	while sub_epoch < 30:
	    if param_updater.check(sub_epoch):
	    	#print(epoch*40 + sub_epoch)
	        ## update which parameters are updated
	        transfer = param_updater.refresh(sub_epoch)
	        print('Updating params: ',epoch*30 + sub_epoch, transfer)
	        ## get a new trainer with different learnable parameters
	        trainer = pipeline.ComposerTrainer( model, train_loader, args.lr, args.lights_mult, args.un_mult, args.lab_mult, transfer, 
	                                    epoch_size=args.epoch_size, iters=args.iters)

	    # foll is uncommented since not using early stopping in this run 
	    if args.save_model:
	        state = model.state_dict()
	        torch.save( state, open(os.path.join(args.save_path, 'state.t7'), 'wb') )

	    
	    ## visualize intrinisc image predictions and reconstructions of the val set
	    val_losses = pipeline.visualize_composer(model, val_loader, args.save_path, epoch)
	    #wound_val_losses = pipeline.visualize_composer_wound(model, wound_val_loader, args.save_path, epoch)
	   
	    ## one sweep through the args.epoch_size images
	    train_losses = trainer.train()
	    
	    ## save training of the errors
	    logger.update(train_losses, val_losses)
	    # logger.update(train_losses, wound_val_losses)
	    # print ("Wound val_losses:", wound_val_losses)

	    # refl_es.check_early_stopping(train_losses[1] ,val_losses[1],model,trainer.optimizer, epoch, args.save_path)
	    # shape_es.check_early_stopping(train_losses[3],val_losses[3],model,trainer.optimizer, epoch, args.save_path)
	    # light_es.check_early_stopping(train_losses[4],val_losses[4],model,trainer.optimizer, epoch, args.save_path)
	    # shad_es.check_early_stopping(train_losses[5],val_losses[5],model,trainer.optimizer, epoch, args.save_path)

	    # if shad_es.early_stop:
	    # 	print ("Shading decoder overfitting. Starting to train refl/normals decoder")
	    # 	sub_epoch = 9

	    # if refl_es.early_stop or shape_es.early_stop:
	    #     print ("Refl/normals decoder overfitting. Starting to train lights decoder")
	    #     sub_epoch = 19

	    # if light_es.early_stop:
	    # 	print ("Lights decoder overfitting. Starting to train shading decoder")
	    # 	ksub_epoch = 39

	    sub_epoch += 1

	    # if refl_es.early_stop and shape_es.early_stop and shad_es.early_stop and light_es.early_stop:
	    # 	print (" all networks overfitting. stopping training")
	    # 	stop = True






