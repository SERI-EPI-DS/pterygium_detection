import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch, torchvision
from torchvision import transforms
from torch.nn import functional as F
from torchvision.models.utils import load_state_dict_from_url


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)




def get_model(task = 'any_ptergium'):
    task_types = ['any_pterygium', 'referable_pterygium']
    assert task in task_types, f"Pick from {task_types}"

    state_dict = load_state_dict_from_url(f'https://github.com/SERI-EPI-DS/pterygium_detection/releases/download/v1.0/{task}.pth')

    # Binary pterygium

    model=torchvision.models.vgg16_bn(num_classes = 2)
    model.load_state_dict(state_dict)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device);
    model.eval();
    return model, device

def get_data_loader(root_dir, num_workers, batch_size):
    image_transformations = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    def test_valid_file(path):
        try:
            _ = Image.open(path)
        except:
            return False
        return True


    asp_dataset = torchvision.datasets.ImageFolder(root_dir,
                                                   is_valid_file = test_valid_file,
                                                   transform= image_transformations)

    data_loader = torch.utils.data.DataLoader(asp_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
    return data_loader

def get_predictions(model, data_loader, device):

    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs=  data[0].to(device)
            preds = model(inputs)
            predictions.append(F.softmax(preds.detach(), dim=1).cpu().numpy())

    predictions=np.concatenate(predictions)

    return predictions



def main(args):

    model, device = get_model(args.task_type)

    dataloader = get_data_loader(args.folder_path, args.workers, args.batch_size)



    predictions = get_predictions(model, dataloader, device)



    files = [i[0].replace(args.folder_path, '') for i in dataloader.dataset.imgs]
    df = pd.DataFrame({'files':files, 'prediction_probability': predictions[:,1]})

    df.to_csv(args.df_save_path, index=False)
    return






if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Store model predictions as a csv')
    args.add_argument('task_type',
                    default='any_pterygium',
                    const='any_pterygium',
                    nargs='?',
                    choices=['any_pterygium', 'referable_pterygium'],
                    help='which model to load: (default: %(default)s)')
    args.add_argument('folder_path', type=str,
                      help='path to folder with images')
    args.add_argument('-w', '--workers', default=6, type=int,
                      help='number of cores to use in parallel')
    args.add_argument('-b', '--batch_size', default=64, type=int,
                      help='batch size')
    args.add_argument('-s', '--df_save_path', default='./predictions.csv', type=str,
                      help='path to save predictions')
    main(args.parse_args())
