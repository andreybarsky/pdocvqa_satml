import os, socket, datetime


import torch
import wandb as wb
import os

import json
from utils import Singleton


class Logger(metaclass=Singleton):

    def __init__(self, config):

        self.log_folder = config.save_dir
        self.experiment_name = config.experiment_name
        self.comms_log_file = os.path.join(self.log_folder, "communication_logs", "{:}.csv".format(self.experiment_name))

        machine_dict = {'cvc117': 'Local', 'cudahpc16': 'DAG', 'cudahpc25': 'DAG-A40'}
        machine = machine_dict.get(socket.gethostname(), socket.gethostname())

        dataset = config.dataset_name
        visual_encoder = getattr(config, 'visual_module', {}).get('model', '-').upper()

        tags = [config.model_name, dataset, machine]
        log_config = {
            'Model': config.model_name, 'Weights': config.model_weights, 'Dataset': dataset,
            'Visual Encoder': visual_encoder, 'Batch size': config.batch_size,
            'Max. Seq. Length': getattr(config, 'max_sequence_length', '-'), 'Learning Rate': config.lr, 'seed': config.seed,
            'No. Training Epoch': config.train_epochs
        }

        if config.use_dp:
            tags.append('DP')
            sampling_prob =  (config.dp_params.providers_per_iteration / config.dp_params.total_providers)
            log_config.update({
                'DP': True,
                'DP Sensitivity': config.dp_params.sensitivity,
                'Noise Multiplier': config.dp_params.noise_multiplier,
                'Client sampling prob.': sampling_prob,
                'Sample Providers': config.dp_params.providers_per_iteration
            })



        if config.lora:
            tags.append('LoRA')
            log_config.update({
                'LoRA': True,
                'LoRA Target Modules': config.lora_params.targets,
                'LoRA Rank': config.lora_params.rank,
                'LoRA Alpha':  config.lora_params.alpha,
                'LoRA Dropout': config.lora_params.dropout
            })

        self.logger = wb.init(project="PDocVQA", name=self.experiment_name, dir=self.log_folder, tags=tags, config=log_config)

        if config.use_dp:
            # create subsample dir:
            self.subsample_dir = os.path.join(self.log_folder, 'run_subsamples', self.logger.name)
            if not os.path.exists(self.subsample_dir):
                os.makedirs(self.subsample_dir)
            self.sampled_providers = {}

        # self.logger.define_metric("Train/FL Round *", step_metric="fl_round")
        # self.logger.define_metric("Val/FL Round *", step_metric="fl_round")
        self._print_config(log_config)

        self.current_epoch = 0
        self.len_dataset = 0

    def _print_config(self, config):
        print("{:s}: {:s} \n{{".format(config['Model'], config['Weights']))
        for k, v in config.items():
            if k != 'Model' and k != 'Weights':
                print("\t{:}: {:}".format(k, v))
        print("}\n")

    def log_model_parameters(self, model):
        total_params = 0
        trainable_params = 0
        for attr in dir(model):
            if isinstance(getattr(model, attr), torch.nn.Module):
                total_params += sum(p.numel() for p in getattr(model, attr).parameters())
                trainable_params += sum(p.numel() for p in getattr(model, attr).parameters() if p.requires_grad)

        self.logger.config.update({
            'Model Params': int(total_params / 1e6),  # In millions
            'Model Trainable Params': int(trainable_params / 1e6)  # In millions
        })

        print("Model parameters: {:d} - Trainable: {:d} ({:2.2f}%)".format(
            total_params, trainable_params, trainable_params / total_params * 100))

    def log_val_metrics(self, accuracy, anls, update_best=False, providers=None):
        str_msg = "Epoch {:d}: Accuracy {:2.2f}     ANLS {:2.4f}".format(self.current_epoch, accuracy*100, anls)

        if self.logger:
            log_dct = {
                'Val/Epoch Accuracy': accuracy,
                'Val/Epoch ANLS': anls,
                'Epoch': self.current_epoch,
            }
            if providers is not None:
                log_dct['Providers'] = providers

            self.logger.log(log_dct)

            if update_best:
                str_msg += "\tBest Accuracy!"
                self.logger.config.update({
                    "Best Accuracy": accuracy,
                    "Best Epoch": self.current_epoch
                }, allow_val_change=True)

        print(str_msg)

    def log_subsample(self):
        # save subsample details at each epoch to file:
        subsample_path = os.path.join(self.subsample_dir, f'subsamples_e{self.current_epoch:d}.json')
        with open(subsample_path, 'w') as f:
            print(f"saving subsample info to: {subsample_path}")
            json.dump(self.sampled_providers, f, indent=2)
