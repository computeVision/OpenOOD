import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle
import collections
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from openood.evaluation_api import Evaluator
from openood.datasets.imglist_dataset import ImglistDataset

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights
from torch.hub import load_state_dict_from_url

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50
from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from openood.evaluation_api.preprocessor import default_preprocessing_dict

from torchvision.utils import make_grid

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from foolbox import PyTorchModel

home = os.getenv('HOME')


class ModelWrapper:
    def __init__(self, model, mean, std):
        self.model = model
        self.normalization = transforms.Normalize(mean, std)

    def normalize_input(self, input_data):
        # Apply the stored normalization to the input data
        normalized_input = self.normalization(input_data)
        return normalized_input

    def __getattr__(self, attr):
        # Delegate attribute access to the original model
        return getattr(self.model, attr)

    def __call__(self, input_data):
        normalized_input = self.normalize_input(input_data)
        return self.model(normalized_input)


def predict(model, input_batch, labels):
    logits = model(input_batch)
    # Get the predicted class label
    predictions = logits.argmax(-1)

    accuracy = (predictions == labels).cpu().float().mean().item()
    print("accuracy", accuracy)
    return predictions


def get_dataloaders(
        id_data, batch_size=200, 
        tvs_pretrained=True, tvs_version=1,
        arch='resnet50'
    ):
    attacks = ['pgd', 'fgsm', 'df', 'masked_pgd']
    base_path_imglist = os.path.join(home, "IWR/OpenOOD/data/benchmark_imglist"

    preprocessor_data = default_preprocessing_dict[id_data]
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 4
    }
    num_classes = 1000

    if id_data in ['cifar10', 'cifar100', 'imagenet200']:
        NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
        MODEL = {
            'cifar10': ResNet18_32x32,
            'cifar100': ResNet18_32x32,
            'imagenet200': ResNet18_224x224,
        }
        arch = 'resnet18'

        try:
            num_classes = NUM_CLASSES[id_data]
            model_arch = MODEL[id_data]
        except KeyError:
            raise NotImplementedError(f'ID dataset {id_data} is not supported.')

        net = model_arch(num_classes=num_classes)

        if id_data == "cifar10":
            imglist_pth = os.path.join(base_path_imglist, "cifar10/test_cifar10.txt")
            imglist_pth_attacks = [
                "cifar10/test_pgd_ResNet18_32x32_cifar10.txt",
                "cifar10/test_fgsm_ResNet18_32x32_cifar10.txt",
                "cifar10/test_df_ResNet18_32x32_cifar10.txt",
                "cifar10/test_masked_pgd_ResNet18_32x32_cifar10.txt"
            ]
            data_dir = os.path.join(home, "IWR/OpenOOD/data/images_classic")
            best_path = os.path.join(home, "IWR/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt")
        elif id_data == "cifar100":
            imglist_pth = os.path.join(home, "IWR/OpenOOD/data/benchmark_imglist/cifar100/test_cifar100.txt")
            imglist_pth_attacks = [
                "cifar100/test_pgd_ResNet18_32x32_cifar100.txt",
                "cifar100/test_fgsm_ResNet18_32x32_cifar100.txt",
                "cifar100/test_df_ResNet18_32x32_cifar100.txt",
                "cifar100/test_masked_pgd_ResNet18_32x32_cifar100.txt"
            ]
            data_dir = os.path.join(home, "IWR/OpenOOD/data/images_classic")
            best_path = os.path.join(home, "IWR/OpenOOD/results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt")
        elif id_data == "imagenet200":
            imglist_pth = os.path.join(home, "IWR/OpenOOD/data/benchmark_imglist/imagenet200/test_imagenet200.txt")
            imglist_pth_attacks = [
                "imagenet200/test_pgd_ResNet18_224x224_imagenet200.txt",
                "imagenet200/test_fgsm_ResNet18_224x224_imagenet200.txt",
                "imagenet200/test_df_ResNet18_224x224_imagenet200.txt",
                "imagenet200/test_masked_pgd_ResNet18_224x224_imagenet200.txt"
            ]
            data_dir = os.path.join(home, "IWR/OpenOOD/data/images_largescale")
            best_path = os.path.join(home, "IWR/OpenOOD/results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2/best.ckpt")
        
        net.load_state_dict(
            torch.load(best_path, map_location='cpu')
            )


    elif id_data == 'imagenet':
        data_dir = os.path.join(home, "IWR/OpenOOD/data/images_largescale")
        imglist_pth = os.path.join(base_path_imglist, "imagenet/test_imagenet.txt")


        if tvs_pretrained:
            if arch == 'resnet50':
                net = ResNet50()
                weights = eval(f'ResNet50_Weights.IMAGENET1K_V{tvs_version}')
                net.load_state_dict(load_state_dict_from_url(weights.url))
                # preprocessor = weights.transforms()
                imglist_pth_attacks = [
                    "imagenet/test_pgd_resnet50_imagenet.txt",
                    "imagenet/test_fgsm_resnet50_imagenet.txt",
                    "imagenet/test_df_resnet50_imagenet.txt",
                    "imagenet/test_masked_pgd_resnet50_imagenet.txt"
                ]
                
            elif arch == 'swin-t':
                net = Swin_T()
                weights = eval(f'Swin_T_Weights.IMAGENET1K_V{tvs_version}')
                net.load_state_dict(load_state_dict_from_url(weights.url))
                # preprocessor = weights.transforms()
                imglist_pth_attacks = [
                    "imagenet/test_pgd_swin-t_imagenet.txt",
                    "imagenet/test_fgsm_swin-t_imagenet.txt",
                    "imagenet/test_df_swin-t_imagenet.txt",
                    "imagenet/test_masked_pgd_swin-t_imagenet.txt"
                ]

            elif arch == 'vit-b-16':
                net = ViT_B_16()
                weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{tvs_version}')
                net.load_state_dict(load_state_dict_from_url(weights.url))
                # preprocessor = weights.transforms()
                imglist_pth_attacks = [
                    "imagenet/test_pgd_vit-b-16_imagenet.txt",
                    "imagenet/test_fgsm_vit-b-16_imagenet.txt",
                    "imagenet/test_df_vit-b-16_imagenet.txt",
                    "imagenet/test_masked_pgd_vit-b-16_imagenet.txt"
                ]

            else:
                raise NotImplementedError

    net.cuda()
    net.eval()

    dataloader_dict = {}
    dataloader_dict['id_data'] = id_data
    dataloader_dict['arch'] = arch
    dataloader_dict['model'] = net
    dataloader_dict['preprocessing'] = preprocessor_data

    if id_data in ["cifar10", "cifar100"]:
        preprocessor = transforms.Compose([
            # transforms.Resize(preprocessor_data["pre_size"]),  # Resize the image to the required input size of your model
            # transforms.CenterCrop(preprocessor_data["img_size"]),
            transforms.ToTensor(),     
            # transforms.Normalize(mean=preprocessor_data["normalization"][0], std=preprocessor_data["normalization"][1]),  
        ])

        dataset_benign = ImglistDataset(
            name=id_data + "_benign",
            imglist_pth=imglist_pth,
            data_dir=data_dir,
            num_classes=num_classes,
            preprocessor=preprocessor,
            data_aux_preprocessor=preprocessor
            )
        dataloader_dict["benign"] = DataLoader(dataset_benign, **loader_kwargs)

        for it, att in enumerate(imglist_pth_attacks):
            imglist_pth_att = os.path.join(base_path_imglist, att)
            dataset_att = ImglistDataset(
                name=id_data + "_" + attacks[it],
                imglist_pth=imglist_pth_att,
                data_dir=data_dir,
                num_classes=num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=preprocessor
                )
            dataloader_dict[attacks[it]] = DataLoader(dataset_att, **loader_kwargs)


    if id_data in ["imagenet200", "imagenet"]:
        preprocessor_benign = transforms.Compose([
            transforms.Resize(preprocessor_data["pre_size"]),
            transforms.CenterCrop(preprocessor_data["img_size"]),
            transforms.ToTensor(),     
            # transforms.Normalize(mean=preprocessor_data["normalization"][0], std=preprocessor_data["normalization"][1]),
        ])

        preprocessor_att = transforms.Compose([
            # transforms.Resize(preprocessor_data["pre_size"]),
            # transforms.CenterCrop(preprocessor_data["img_size"]),
            transforms.ToTensor(),     
            # transforms.Normalize(mean=preprocessor_data["normalization"][0], std=preprocessor_data["normalization"][1]),
        ])

        dataset_benign = ImglistDataset(
            name=id_data + "_benign",
            imglist_pth=imglist_pth,
            data_dir=data_dir,
            num_classes=num_classes,
            preprocessor=preprocessor_benign,
            data_aux_preprocessor=preprocessor_benign
            )
        dataloader_dict["benign"] = DataLoader(dataset_benign, **loader_kwargs)

        for it, att in enumerate(imglist_pth_attacks):
            imglist_pth_att = os.path.join(base_path_imglist, att)
            dataset_att = ImglistDataset(
                name=id_data + "_" + attacks[it],
                imglist_pth=imglist_pth_att,
                data_dir=data_dir,
                num_classes=num_classes,
                preprocessor=preprocessor_att,
                data_aux_preprocessor=preprocessor_att
                )
            dataloader_dict[attacks[it]] = DataLoader(dataset_att, **loader_kwargs)


    return dataloader_dict


def calc_xai(dataloaders):
    """
    https://github.com/jacobgil/pytorch-grad-cam
    """

    net = dataloaders['model']
    mean = dataloaders['preprocessing']['normalization'][0]
    std = dataloaders['preprocessing']['normalization'][1]
    print(dataloader['id_data'], dataloader['arch'], mean, std)

    model = ModelWrapper(net, mean=mean, std=std)
    
    # preprocessing = dict(mean=mean, std=std, axis=-3)
    # fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    if dataloaders['arch'] in ['resnet18', 'resnet50']:
        target_layers = [model.layer4[-1]]
    elif dataloaders['arch'] == 'swin-t':
        # https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/swinT_example.py
        target_layers = [model.features[-1][-1].norm1]
    elif dataloaders['arch'] == 'vit-b-16':
        # https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py
        target_layers = [model.encoder.layers[-1]]

    # Create normalization transform
    # normalize = transforms.Normalize(mean=dataloaders['preprocessing']['normalization'][0], std=dataloaders['preprocessing']['normalization'][1])
    
    def iter_next(iterator):
        inputs = next(iterator)
        data = inputs["data"].cuda()
        labels = inputs["label"].cuda()
        return data, labels

    iter_loader_pgd  = iter(dataloaders['pgd'])
    iter_loader_fgsm = iter(dataloaders['fgsm'])
    iter_loader_df   = iter(dataloaders['df'])
    iter_loader_mpgd = iter(dataloaders['masked_pgd'])
    
    predictions = {}
    xaimaps = {}

    xaimaps_ben  = []
    xaimaps_pgd  = []
    xaimaps_fgsm = []
    xaimaps_df   = []
    xaimaps_mpgd = []

    predictions_gt   = []
    predictions_ben  = []
    predictions_pgd  = []
    predictions_fgsm = []
    predictions_df   = []
    predictions_mpgd = []

    targets = None # [ClassifierOutputTarget(None)]
    with GradCAM(model=model, target_layers=target_layers) as cam:

        for data_ben in dataloaders['benign']:
            data = data_ben["data"].cuda()
            labels = data_ben["label"].cuda()

            data_pgd, labels_pgd = iter_next(iter_loader_pgd)
            data_fgsm, labels_fgsm = iter_next(iter_loader_fgsm)
            data_df, labels_df = iter_next(iter_loader_df)
            data_mpgd, labels_mpgd = iter_next(iter_loader_mpgd)

            outputs_ben  = cam(input_tensor=data,      targets=targets)
            outputs_pgd  = cam(input_tensor=data_pgd,  targets=targets)
            outputs_fgsm = cam(input_tensor=data_fgsm, targets=targets)
            outputs_df   = cam(input_tensor=data_df,   targets=targets)
            outputs_mpgd = cam(input_tensor=data_mpgd, targets=targets)

            # Append xaimaps to the list
            xaimaps_ben.append(outputs_ben)
            xaimaps_pgd.append(outputs_pgd)
            xaimaps_fgsm.append(outputs_fgsm)
            xaimaps_df.append(outputs_df)
            xaimaps_mpgd.append(outputs_mpgd)

            predicted_ben  = predict(model, data, labels)
            predicted_pgd  = predict(model, data_pgd, labels)
            predicted_fgsm = predict(model, data_fgsm, labels)
            predicted_df   = predict(model, data_df, labels)
            predicted_mpgd = predict(model, data_mpgd, labels)

            # Append predictions to the list
            predictions_gt.append(labels.cpu())
            predictions_ben.append(predicted_ben.cpu())
            predictions_pgd.append(predicted_pgd.cpu())
            predictions_fgsm.append(predicted_fgsm.cpu())
            predictions_df.append(predicted_df.cpu())
            predictions_mpgd.append(predicted_mpgd.cpu())

    xaimaps["ben"] = np.concatenate(xaimaps_ben)
    xaimaps["pgd"] = np.concatenate(xaimaps_pgd)
    xaimaps["fgsm"] = np.concatenate(xaimaps_fgsm)
    xaimaps["df"] = np.concatenate(xaimaps_df)
    xaimaps["mpgd"] = np.concatenate(xaimaps_mpgd)
    
    predictions["gt"] = torch.cat(predictions_gt)
    predictions["ben"] = torch.cat(predictions_ben)
    predictions["pgd"] = torch.cat(predictions_pgd)
    predictions["fgsm"] = torch.cat(predictions_fgsm)
    predictions["df"] = torch.cat(predictions_df)
    predictions["mpgd"] = torch.cat(predictions_mpgd)

    return xaimaps, predictions


if __name__ == "__main__":
    dataloader = get_dataloaders(id_data="cifar10", batch_size=200)
    gradcams, predictions = calc_xai(dataloader)
    torch.save(gradcams,    os.path.join(home, "IWR/OpenOOD/results/xai/gradcams_cifar10.pt"))
    torch.save(predictions, os.path.join(home, "IWR/OpenOOD/results/xai/predictions_cifar10.pt"))

    # dataloader = get_dataloaders(id_data="cifar100", batch_size=200)
    # gradcams, predictions = calc_xai(dataloader)
    # torch.save(gradcams, os.path.join(home, "IWR/OpenOOD/results/xai/gradcams_cifar100.pt"))
    # torch.save(predictions, os.path.join(home, "IWR/OpenOOD/results/xai/predictions_cifar100.pt"))

    # dataloader = get_dataloaders(id_data="imagenet200", batch_size=64)
    # gradcams, predictions = calc_xai(dataloader)
    # torch.save(gradcams,    os.path.join(home, "IWR/OpenOOD/results/xai/gradcams_imagenet200.pt"))
    # del gradcams
    # torch.save(predictions, os.path.join(home, "IWR/OpenOOD/results/xai/predictions_imagenet200.pt"))
    # del predictions

    # dataloader = get_dataloaders(id_data="imagenet", batch_size=32)
    # gradcams, predictions = calc_xai(dataloader)
    # torch.save(gradcams, os.path.join(home, "IWR/OpenOOD/results/xai/gradcams_imagenet.pt", pickle_protocol=5))
    # del gradcams
    # torch.save(predictions, os.path.join(home, "IWR/OpenOOD/results/xai/predictions_imagenet.pt"))
    # del predictions

    # dataloader = get_dataloaders(id_data="imagenet", batch_size=32, arch="swin-t")
    # gradcams, predictions = calc_xai(dataloader)
    # torch.save(gradcams, os.path.join(home, "IWR/OpenOOD/results/xai/gradcams_imagenet_swint.pt", pickle_protocol=5))
    # del gradcams
    # torch.save(predictions, os.path.join(home, "IWR/OpenOOD/results/xai/predictions_imagenet_swint.pt"))
    # del predictions

    # dataloader = get_dataloaders(id_data="imagenet", batch_size=32, arch="vit-b-16")
    # gradcams, predictions = calc_xai(dataloader)
    # torch.save(gradcams, os.path.join(home, "IWR/OpenOOD/results/xai/gradcams_imagenet_vitb16.pt", pickle_protocol=5))
    # del gradcams
    # torch.save(predictions, os.path.join(home, "IWR/OpenOOD/results/xai/predictions_imagenet_vitb16.pt"))
    # del predictions

    print("done")