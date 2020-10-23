from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from base import BaseDataLoader
from torchvision import transforms, utils
from PIL import Image

class ImageDataset(Dataset):
	"""
	Dataset reads csv and generates input data from image and specified column.
	"""
	def __init__(self,
				 dataframe_location,
				 label_variable,
				 path_variable='dir',
				 images_root_dir=None,
				 class_mode=None,
				 n_channels=3,
				 return_path_also=False,
				 transform=None):
		'''
		Inputs:
		- dataframe_location, images_root_dir: give image paths
		- data_variable: column in dataframe which has Y values
		- class_mode: Same as keras ImageDataGenerator class mode. None, 'sparse', 'categorical'. Use sparse for CrossEntropyLoss.
		- n_channels: image colour channels. Use 3 for RGB.
		- return_path_also: return image paths to check. Used during error analysis
		- transform: torchvision image trasnforms
		'''
		self.dataset=pd.read_csv(dataframe_location)
		self.root_dir=images_root_dir
		self.n_channels = n_channels
		assert label_variable in self.dataset.columns, "Label variable specified is not present in dataset."
		self.label_variable=label_variable
		self.transform=transform
		self.path_variable=path_variable
		self.class_mode=None if class_mode=="None" else class_mode # As None gets converted to "None" in json
		self.return_path_also=return_path_also
		if self.class_mode:
			self.classes=sorted(list(self.dataset[label_variable].unique()))
			self.classes_map=dict(zip(self.classes, range(len(self.classes))))
		else:
			self.classes_map=None

	def __getitem__(self, idx):
		'Generate one batch of data'
		if torch.is_tensor(idx):
			idx = idx.tolist()
		if isinstance(idx, list):
			idx=idx[0]
		filename=self.dataset.iloc[idx][self.path_variable]

		if self.root_dir is not None:
			filename=os.path.join(self.root_dir, filename)

		if self.n_channels == 3:
			img=Image.open(filename).convert('RGB')
		elif self.n_channels == 1:
			img=Image.open(filename).convert('L')
		else:
			raise NameError("Use OCTVolume dataset for multi channel images.")

		if self.transform is not None:
			try:
				img = self.transform(img)
			except OSError:
				print(filename)
		target=self.dataset.iloc[idx][self.label_variable]

		if self.classes_map:
			target=self.classes_map[target]
			if self.class_mode=="categorical":
				target = np.eye(len(self.classes_map))[target]

		if self.return_path_also:
			file_path=self.dataset.iloc[idx][self.path_variable]
			sample = (img, target, file_path)
		else:
			sample = (img, target)
		return sample

	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.dataset)

class ImageDataLoader(BaseDataLoader):
	"""
	Generic data loader for reading images and labels from a dataframe
	"""
	def __init__(self, dataset_location, root_dir_images,
	label_variable, path_variable, input_shape, input_channels,
	batch_size, shuffle=True, validation_split=0.0,
	num_workers=1, class_mode=None,
	rotate_degrees=0, translate_fraction=None, shear_degrees=None, scale_range=None,
	horizontal_flip=False, vertical_flip=False):

		transform_list=[]

		if horizontal_flip:
			transform_list.append(transforms.RandomHorizontalFlip())
		if vertical_flip:
			transform_list.append(transforms.RandomVerticalFlip())
		if rotate_degrees>0 or translate_fraction or scale_range or shear_degrees:
			transform_list.append(transforms.RandomAffine(rotate_degrees,
														  translate=tuple(translate_fraction),
														  scale=tuple(scale_range),
														  shear=shear_degrees))
		transform_list.append(transforms.Resize(input_shape))
		transform_list.append(transforms.ToTensor())
		if input_channels==1:
			transform_list.append(transforms.Normalize((0.5,), (0.5,)))
		else:
			transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5] ))

		dataset_kwargs={
			"dataframe_location":dataset_location,
			"label_variable":label_variable,
			"path_variable":path_variable,
			"images_root_dir":root_dir_images,
			"class_mode":class_mode,
			"n_channels":input_channels
		}
		print(transforms.Compose(transform_list))
		self.train_dataset = ImageDataset(**dataset_kwargs, transform=transforms.Compose(transform_list))
		super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers)
