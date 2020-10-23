import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models

def initialize_model(model_name, n_channels, num_classes, use_pretrained=True):
	"""
	Initialize these variables which will be set in this if statement. Each of these variables is model specific.
	If feature_extract = False, the model is finetuned and all model parameters are updated.
	If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
	"""
	model_ft = None
	input_size = 0
	if model_name.startswith("resnet"):
		""" Resnet18, Resnet34, Resnet50, Resnet101, Resnet152
		"""
		model_ft = getattr(models, model_name)(pretrained=use_pretrained)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name.startswith("vgg"):
		""" VGG11, VGG11_BN, VGG13,  VGG13_BN, VGG16,  VGG16_BN, VGG19,  VGG19_BN
		"""
		model_ft = getattr(models, model_name)(pretrained=use_pretrained)
		num_ftrs = model_ft.classifier[0].in_features
		latent_ftrs = model_ft.classifier[0].out_features
		model_ft.classifier = nn.Sequential(
			nn.Linear(num_ftrs,latent_ftrs),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(latent_ftrs,latent_ftrs),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(latent_ftrs,num_classes)
		)
		input_size = 224

	elif model_name.startswith("densenet"):
		""" Densenet121, Densenet161, Densenet169, Densenet201
		"""
		model_ft = getattr(models, model_name)(pretrained=use_pretrained)
		num_ftrs = model_ft.classifier.in_features
		model_ft.classifier = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name.startswith("mobilenet"):
		""" mobilenet_v2
		"""
		model_ft = getattr(models, model_name)(pretrained=use_pretrained)
		num_ftrs = model_ft.classifier[1].in_features
		model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	return model_ft, input_size
