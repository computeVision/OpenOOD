from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as trn
from tqdm import tqdm
import time
from datetime import datetime

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.networks.scale_net import ScaleNet

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor

from openood.attacks.misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir,
    str2bool,
    create_log_file,
    save_log
)

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa

DEEPFOOL   = ['fgsm', 'bim', 'pgd', 'df', 'cw'] + ['pgd_bpda']
AUTOATTACK = ['aa', 'apgd-ce', 'square']

class AttackDataset:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Callable = None,
        normalize: Callable = None,
        # postprocessor_name: str = None,
        # postprocessor: Type[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # # check the arguments
        # if postprocessor_name is None and postprocessor is None:
        #     raise ValueError('Please pass postprocessor_name or postprocessor')
        # if postprocessor_name is not None and postprocessor is not None:
        #     print(
        #         'Postprocessor_name is ignored because postprocessor is passed'
        #     )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # get postprocessor
        # if postprocessor is None:
        #     postprocessor = get_postprocessor(config_root, postprocessor_name,
        #                                       id_name)
        # if not isinstance(postprocessor, BasePostprocessor):
        #     raise TypeError(
        #         'postprocessor should inherit BasePostprocessor in OpenOOD')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, att=True, **loader_kwargs)

        # # wrap base model to work with certain postprocessors
        # if postprocessor_name == 'react':
        #     net = ReactNet(net)
        # elif postprocessor_name == 'ash':
        #     net = ASHNet(net)
        # elif postprocessor_name == 'scale':
        #     net = ScaleNet(net)

        # postprocessor setup
        # postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])

        self.id_name = id_name
        self.data_root = data_root
        self.net = net
        self.normalize = normalize
        self.preprocessor = preprocessor
        # self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # # perform hyperparameter search if have not done so
        # if (self.postprocessor.APS_mode
        #         and not self.postprocessor.hyperparam_search_done):
        #     self.hyperparam_search()

        self.net.eval()


    def run_attack(self, args):
        self.net.eval()

        preprocessing = dict(mean=self.normalize['mean'], std=self.normalize['std'], axis=-3)
        fmodel = PyTorchModel(self.net, bounds=(0, 1), preprocessing=preprocessing)

        if args.att == 'fgsm':
            attack = fa.FGSM()
        elif args.att == 'pgd':
            attack = fa.LinfPGD()
        # elif args.att == 'eot_pgd':
        #     from utils.modelwrapper import ModelWrapper
        #     wrapped_model = ModelWrapper(model)
        #     attack = ta.EOTPGD(wrapped_model, args.eps, alpha=2/255, steps=10, eot_iter=2) # https://github.com/xuanqing94/BayesianDefense/blob/master/train_adv.sh
        elif args.att == 'df':
            attack = fa.L2DeepFoolAttack()
            args.eps = None
        elif args.att == 'cw':
            attack = fa.L2CarliniWagnerAttack(steps=1000)
            args.eps = None
        # elif args.att in AUTOATTACK:
        #     # from helper_gen.attacks.sub_autoattack.auto_attack import AutoAttack as AutoAttack_mod
        #     from whitebox_attacks.autoattack.autoattack import AutoAttack as AutoAttack_mod
        #     # https://colab.research.google.com/drive/1uZrW3Sg-t5k6QVEwXDdjTSxWpiwPGPm2?usp=sharing#scrollTo=jYnKIzXAgV4W
        #     adversary = AutoAttack_mod(fmodel, norm=args.norm.capitalize(), eps=args.eps, 
        #                                 log_path=os.path.join(log_pth, args.load_json.split('/')[-1]).replace("json", "log"),  verbose=False, version=args.version)
        #     if args.version == 'custom':
        #         adversary.attacks_to_run = [ args.att ]
        #     adversary.seed = 0 # every attack is seeded by 0. Otherwise, it would be randomly seeded for each attack.
        elif args.att == 'masked_pgd':
            from openood.attacks import masked_pgd_attack, NormalizeWrapper
            norm_model = NormalizeWrapper(self.net, self.normalize['mean'], self.normalize['std'])

        base_pth = os.path.join('./data/attacked', args.att + "_" + self.id_name + "_" + args.arch)
        create_dir(base_pth)
        log_pth = os.path.join(base_pth, 'logs')
        log = create_log_file(args, log_pth)
        log['timestamp_start'] =  datetime.now().strftime("%Y-%m-%d-%H:%M")

        attack_path = os.path.join(self.data_root, 'images_largescale', 'imagenet_1k' + '_' + args.att + '_' + args.arch, 'val')
        create_dir(attack_path)

        start_time = time.time()

        counter = 24234
        total_samples = 0
        correct_predicted = 0
        successful_attacked = 0

        try:
            # with torch.no_grad():
            for batch in tqdm(self.dataloader_dict['id']['test'], desc="attack", disable=not True):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                logits = fmodel(data)
                preds = logits.argmax(1)

                total_samples += len(label)
                correct_predicted += (label==preds).cpu().sum().item()

                if args.att in DEEPFOOL:
                    raw_advs, clipped_advs, success = attack(fmodel, data, label, epsilons=args.eps)

                # if args.att in AUTOATTACK:
                #     if args.version == 'standard':
                #         clipped_advs, y_, max_nr, success = adversary.run_standard_evaluation(data, label, bs=args.bs, return_labels=True)
                #     else: 
                #         adv_complete = adversary.run_standard_evaluation_individual(data, label, bs=args.bs, return_labels=True)
                #         clipped_advs, y_, max_nr, success = adv_complete[ args.att ]  
                
                if args.att == 'masked_pgd':
                    clipped_advs, success = masked_pgd_attack(norm_model, data, label, epsilon=1, alpha=0.01, num_steps=40, patch_size=60)
                
                if args.att == 'eot_pgd':
                    clipped_advs = attack(data, label)
                    pred = torch.max(fmodel(clipped_advs),dim=1)[1]
                    success = ~(pred == label)

                if args.att in ['bandits', 'nes']:
                    from blackbox_attacks.bandits import make_adversarial_examples as bandits_attack
                    out = bandits_attack(data, label, args, fmodel, 256)
                    success = out['success_adv']
                    clipped_advs = out['images_adv']
                
                success = success.cpu()
                successful_attacked += success.sum().item()

                for it, suc in enumerate(success):
                    counter += 1 
                    clipped_adv = clipped_advs[it].cpu()
                    # img_benign = img_batch[it].cpu()
                    # label = lab_batch[it].cpu().item()

                    image_pil_adv = trn.ToPILImage()(clipped_adv)
                    image_pil_adv.save(os.path.join(attack_path, f'ILSVRC2012_val_{counter:08d}.png'))


        except Exception as e:
            print("An exception occurred:", str(e))

        # finally:
        asr = total_samples / successful_attacked

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        # Convert elapsed time to hours and minutes
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)

        log['elapsed_time'] = str(hours) + "h:" + str(minutes) + "m"
        log['total_samples'] = total_samples
        log['correct_predicted'] = correct_predicted
        log['successful_attacked'] = successful_attacked
        log['model_accuracy'] = round(correct_predicted/total_samples,4)
        log['asr'] = round(asr,4)
        # log['clean_acc'] = 0 if len(clean_acc_list) == None else round(np.mean(clean_acc_list), 4)

        save_log(args, log, log_pth)
