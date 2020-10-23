import torch
import torch.nn.functional as F
from .heatmap_utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer, find_mobilenet_layer


class AllCAM(object):
	"""Calculate GradCAM, GradCAM++ saliency maps.

	A simple example:

		# initialize a model, model_dict and gradcam
		resnet = torchvision.models.resnet101(pretrained=True)
		resnet.eval()
		model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
		gradcam = GradCAM(model_dict)

		# get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
		img = load_img()
		normed_img = normalizer(img)

		# get a GradCAM saliency map on the class index 10.
		mask, logit = gradcam(normed_img, class_idx=10)

		# make heatmap from mask and synthesize saliency map using heatmap and img
		heatmap, cam_result = visualize_cam(mask, img)


	Args:
		model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
		verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
	"""
	def __init__(self, model_dict, verbose=False):
		model_type = model_dict['type']
		layer_name = model_dict['layer_name']
		self.model_arch = model_dict['arch'].cuda()

		self.gradients = dict()
		self.activations = dict()
		def backward_hook(module, grad_input, grad_output):
			self.gradients['value'] = grad_output[0]
			return None
		def forward_hook(module, input, output):
			self.activations['value'] = output
			return None

		if 'vgg' in model_type.lower():
			target_layer = find_vgg_layer(self.model_arch, layer_name)
		elif 'resnet' in model_type.lower():
			target_layer = find_resnet_layer(self.model_arch, layer_name)
		elif 'densenet' in model_type.lower():
			target_layer = find_densenet_layer(self.model_arch, layer_name)
		elif 'alexnet' in model_type.lower():
			target_layer = find_alexnet_layer(self.model_arch, layer_name)
		elif 'squeezenet' in model_type.lower():
			target_layer = find_squeezenet_layer(self.model_arch, layer_name)
		elif 'mobilenet' in model_type.lower():
			target_layer = find_mobilenet_layer(self.model_arch, layer_name)
		target_layer.register_forward_hook(forward_hook)
		target_layer.register_backward_hook(backward_hook)

		if verbose:
			try:
				input_size = model_dict['input_size']
			except KeyError:
				print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
				pass
			else:
				device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
				self.model_arch(torch.zeros(1, 3, *(input_size), device=device))


	def forward(self, input, class_idx=None, retain_graph=False):
		"""
		Args:
			input: input image with shape of (1, 3, H, W)
			class_idx (int): class index for calculating GradCAM.
					If not specified, the class index that makes the highest model prediction score will be used.
		Returns:
			cam_dict: dictionary with the following saliency maps
				gradcam
				gradcam++
			logit: model output
		"""
		b, c, h, w = input.size()

		logit = self.model_arch(input)
		if class_idx is None:
			predicted_class = logit.max(1)[-1]
			score = logit[:, logit.max(1)[-1]].squeeze()
		else:
			predicted_class = torch.LongTensor([class_idx])
			score = logit[:, class_idx].squeeze()

		self.model_arch.zero_grad()
		score.backward(retain_graph=retain_graph)
		gradients = self.gradients['value'].cuda()
		activations = self.activations['value'].cuda()

		cam_results={
			'gradcam' : self._gc_map(gradients, activations, h, w),
			'gradcam++' : self._gc_pp_map(gradients, activations, h, w, score),
		}
		return  cam_results, logit

	def _gc_map(self, gradients, activations, h, w):
		b, k, u, v = gradients.size()
		alpha = gradients.view(b, k, -1).mean(2)
		#alpha = F.relu(gradients.view(b, k, -1)).mean(2)
		weights = alpha.view(b, k, 1, 1)

		saliency_map = (weights*activations).sum(1, keepdim=True)
		saliency_map = F.relu(saliency_map)
		saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
		saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
		saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

		return saliency_map

	def _gc_pp_map(self, gradients, activations, h, w, score):
		b, k, u, v = gradients.size()

		alpha_num = gradients.pow(2)
		alpha_denom = gradients.pow(2).mul(2) + \
				activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
		alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

		alpha = alpha_num.div(alpha_denom+1e-7)
		positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
		weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

		saliency_map = (weights*activations).sum(1, keepdim=True)
		saliency_map = F.relu(saliency_map)
		saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
		saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
		saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

		return saliency_map

	def __call__(self, input, class_idx=None, retain_graph=False):
		return self.forward(input, class_idx, retain_graph)
