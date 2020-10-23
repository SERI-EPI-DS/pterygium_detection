import os

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from .heatmap import AllCAM, visualize_cam
from tqdm import trange
'''
| Visualisation type              | Arguments        |
|---------------------------------|------------------|
| GradCAM                         | gradcam          |
| GradCAM++                       | gradcampp        |
'''
def get_heatmap_tensors(imgs, model, config, checkpoint_dir, epoch, visualizations=None, save_images_to_dir=False):

	oct_model_dict=get_model_dict(config)
	oct_model_dict['arch']=model
	gradcam = AllCAM(oct_model_dict, True)

	gradcam_grid=[]
	gradcam_pp_grid=[]

	for tq, img in zip(trange(len(imgs), desc='Heatmaps ', leave=False), imgs):
		cam_results, _ = gradcam(torch.unsqueeze(img, 0).cuda())
		gradcam_grid.append(visualize_cam(cam_results['gradcam'].detach().cpu(), denormalise_img(img))[1])
		gradcam_pp_grid.append(visualize_cam(cam_results['gradcam++'].detach().cpu(), denormalise_img(img))[1])

	# Save heatmaps
	if save_images_to_dir:
		save_path=str(checkpoint_dir).replace('models','heatmaps')
		os.makedirs(save_path, exist_ok=True)
		save_image(gradcam_grid, os.path.join(save_path,f'gc_{epoch:02d}.jpg'))
		save_image(gradcam_pp_grid, os.path.join(save_path,f'gc_pp_{epoch:02d}.jpg'))

	return torch.stack(gradcam_grid), torch.stack(gradcam_pp_grid)

def denormalise_img(img):
	return (img+1)/2.

def get_model_dict(config):
	layername_lookup={
	'alexnet':'features_11',
	'vgg':'features_29',
	'resnet':'layer4',
	'densenet':'features_norm5',
	'squeezenet':'features_12_expand3x3_activation',
	'mobilenet':'features_18_1'
	}
	'''
	Original architectures:
	alexnet
	vgg16
	resnet101
	densenet161
	squeezenet1_1
	'''
	model_dict={}
	model_dict['type']=''.join([i for i in config['arch']['type'] if i.isalpha()])
	for base_model_name in layername_lookup.keys():
		if str(model_dict['type']).startswith(base_model_name):
			model_dict['layer_name']=layername_lookup[base_model_name]
	if 'data_loader' in config.config:
		model_dict['input_size']=tuple(config['data_loader']['args']['input_shape'])
	else:
		model_dict['input_size']=tuple(config['val_loader']['args']['input_shape'])


	return model_dict
