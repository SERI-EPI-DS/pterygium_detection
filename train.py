import argparse
import collections
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import data_loader as module_data
import model.metric as module_metric
from model.model import initialize_model
from torchvision import models  as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('val_loader', module_data)

    # build model architecture, then print to console
    model_name = config.config["arch"]["type"]
    config_input_shape = None
    config_n_channels = None

    config_input_shape = config.config["data_loader"]["args"]["input_shape"]
    config_n_channels = config.config["data_loader"]["args"]["input_channels"]
    model = None

    if (config.config["arch"]["args"]["pretrained"] is True):
        model, input_size = initialize_model(model_name = model_name,
                                            n_channels = config.config["data_loader"]["args"]["input_channels"],
                                            num_classes = config.config["arch"]["args"]["num_classes"],
                                            use_pretrained=config.config["arch"]["args"]["pretrained"])
        assert config_input_shape == [input_size, input_size], f"Wrong sized inputs. Correct input size {input_size}"
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    if 'class_weights' in config.config:
        if config.config['class_weights']=='weighted':
            df = pd.read_csv(config.config['data_loader']['args']['dataset_location'])
            class_weight = df[config.config['data_loader']['args']['label_variable']].count()/df[config.config['data_loader']['args']['label_variable']].value_counts().sort_index()
            class_weight = list(class_weight/sum(class_weight))
            class_weight = [i*len(class_weight) for i in class_weight]
        else:
            class_weight = config.config['class_weights']
        logger.info("Class weights: " + str(class_weight))
        criterion = getattr(nn, config['loss'])(weight=torch.tensor(class_weight).cuda())
    else:
        criterion = getattr(nn, config['loss'])()
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
      config=config,
      data_loader=data_loader,
      valid_data_loader=valid_data_loader,
      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
