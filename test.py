import argparse, os, random
import numpy as np
import torch
from tqdm import tqdm
import data_loader as module_data
import torch.nn as nn
import model.metric as module_metric
from model.model import initialize_model
from parse_config import ConfigParser
import matplotlib.pyplot as plt
from utils import get_confusion_matrix_figure, get_auc_roc_curve, get_heatmap_tensors
from torchvision.utils import save_image

def main(config, weights_paths, save_vis):
	logger = config.get_logger('test')

	# setup data_loader instances
	# data_loader = getattr(module_data, config['data_loader']['type'])(
	#     config['data_loader']['args']['data_dir'],
	#     batch_size=32,
	#     shuffle=False,
	#     validation_split=0.0,
	#     num_workers=6
	# )
	data_loader = config.init_obj('data_loader', module_data)
	# build model architecture
	model_name = config.config["arch"]["type"]
	model, input_size = initialize_model(model_name = model_name,
										n_channels = config.config["data_loader"]["args"]["input_channels"],
										num_classes = config.config["arch"]["args"]["num_classes"],
										use_pretrained=False)
	logger.info(model)

	# get function handles of loss and metrics
	loss_fn = getattr(nn, config['loss'])()
	metric_fns = [getattr(module_metric, met) for met in config['metrics']]

	logger.info('Loading checkpoint: {} ...'.format(config.resume))

	state_dict = torch.load(weights_paths)

	if config['n_gpu'] > 1:
		model = torch.nn.DataParallel(model)
	try:
		model.load_state_dict(state_dict)
		logger.info('Loaded state_dict from {}'.format(os.path.abspath(weights_paths)))
	except RuntimeError as e:
		print("Differnt model architecture in config and state dict path")
		exit()

	# prepare model for testing
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	model.eval()

	total_loss = 0.0
	total_metrics = torch.zeros(len(metric_fns))

	os.makedirs(save_vis, exist_ok = True)
	y=[]
	y_hat=[]
	with torch.no_grad():
		for i, (data, target) in enumerate(tqdm(data_loader)):
			data, target = data.to(device), target.to(device)
			output = model(data)
			# computing loss, metrics on test set
			loss = loss_fn(output, target)
			batch_size = data.shape[0]
			total_loss += loss.item() * batch_size
			for i, metric in enumerate(metric_fns):
				total_metrics[i] += metric(output, target) * batch_size
			y.append(target.cpu().numpy())
			y_hat.append(output.detach().cpu().numpy())


	y=np.concatenate(y)
	y_hat=np.concatenate(y_hat)
	do_visualization(y, y_hat, labels = data_loader.dataset.classes, save_vis = save_vis)
	store_heatmaps(config, model, data_loader, save_vis)

	n_samples = len(data_loader.sampler)
	log = {'loss': total_loss / n_samples}
	log.update({
		met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
	})
	logger.info(log)

def do_visualization(y, y_hat, labels, save_vis):
	roc = get_auc_roc_curve(y, y_hat, len(labels), labels=labels)
	plt.savefig(os.path.join(save_vis , "roc_curve.jpg"))
	plt.close()
	cm = get_confusion_matrix_figure(y, y_hat,  len(labels), labels=labels)
	plt.savefig(os.path.join(save_vis , "confusion_matrix.jpg"))
	plt.close()
	return

def store_heatmaps(config, model, data_loader, save_vis):
	'Store "batch_size" number of images randomly'
	if data_loader.dataset.__len__() > data_loader.batch_size:
		heatmap_sample_indices = sorted(random.sample(range(data_loader.dataset.__len__()), data_loader.batch_size))
	else:
		heatmap_sample_indices = range(data_loader.dataset.__len__())

	images=[]
	targets=[]
	for idx in heatmap_sample_indices:
		img,label = data_loader.dataset.__getitem__(idx)
		images.append(img)
		targets.append(label)
	heatmap_save_dir = os.path.join(save_vis, 'heatmaps/')
	_, _ = get_heatmap_tensors(images, model,
								config, heatmap_save_dir,
								0, save_images_to_dir=True)
	save_image(images, os.path.join(heatmap_save_dir,'inputs.jpg'), normalize=True)
	return

if __name__ == '__main__':
	args = argparse.ArgumentParser(description='Validate models')

	args.add_argument('weights', default=None, type=str,
					  help='path to state dict .pth file (default: None)')
	args.add_argument('-c', '--config', default=None, type=str,
					  help='config file path (default: None)')
	args.add_argument('-d', '--device', default=None, type=str,
					  help='indices of GPUs to enable (default: all)')
	args.add_argument('-s', '--save_vis', default='./saved/validation_visualizations/', type=str,
					  help='path to save visualizations (default: None)')
	args.add_argument('-r', '--resume', default=None, type=str,
					  help='path to latest checkpoint; do not use this flag if using state dict')
	config = ConfigParser.from_args(args)
	main(config, args.parse_args().weights, args.parse_args().save_vis)
