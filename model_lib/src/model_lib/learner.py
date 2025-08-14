import os
import typing
import tqdm
import argparse
import shutil

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as torch_nn
import torch.utils.data as torch_data

import model_lib.model as model_model
import ml.models.utils as ml_utils


class TrainingConfig():
    def __init__(self,
                    model_dir: str,
                    batch_size: int = 16,
                    learning_rate: float = 2e-4,
                    max_grad_norm: float = 1e9,
                    epoch_save_interval: int = 10,
                    max_steps: typing.Optional[int] = 1024):
            self.model_dir = model_dir
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.max_grad_norm = max_grad_norm
            self.epoch_save_interval = epoch_save_interval
            self.max_steps = max_steps

    @staticmethod
    def add_arg_opt(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("TrainingConfig")
        group.add_argument('--model_dir', type=str, required=True)
        group.add_argument('--batch_size', type=int, default=16)
        group.add_argument('--learning_rate', type=float, default=2e-4)
        group.add_argument('--max_grad_norm', type=float, default=1e9)
        group.add_argument('--epoch_save_interval', type=int, default=10)
        group.add_argument('--max_steps', type=int, default=1024)

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "TrainingConfig":
        return TrainingConfig(
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            epoch_save_interval=args.epoch_save_interval,
            max_steps=args.max_steps
        )

def train(override: bool,
          backup: bool,
          dataset: torch_data.Dataset,
          model_config: model_model.UnconditionalConfig,
          training_config: TrainingConfig):

        if override:
            if backup:
                ml_utils.backup_folder(training_config.model_dir)
            else:
                shutil.rmtree(training_config.model_dir)

        device = ml_utils.get_available_device()
        model = model_model.DiffWave(model_config).to(device)
        # train_impl(0, model, dataset, args, params)

        opt = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

        learner = DiffWaveLearner(model, dataset, opt, training_config)
        learner.restore()
        learner.train(device=device)


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return { k: _nested_map(v, map_fn) for k, v in struct.items() }
    return map_fn(struct)

class DiffWaveLearner:
    def __init__(self,
                 model: model_model.DiffWave,
                 dataset: torch_data.Dataset,
                 optimizer: torch.optim.Optimizer,
                 training_config: TrainingConfig,
                 loss_fn = torch_nn.L1Loss()):

        os.makedirs(training_config.model_dir, exist_ok=True)
        self.model_dir = training_config.model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.config = training_config
        self.loss_fn = loss_fn

        self.epoch = 0
        self.losses=[]

        self.noise_level = self.model.get_noise_level()
        self.summary_writer = None

        self.restore()

    def save(self, filename='weights'):
        save_basename = f'{filename}-{self.epoch}.pt'
        save_name = os.path.join(self.model_dir, save_basename)
        link_name = os.path.join(self.model_dir, f'{filename}.pt')

        model_state = {
                'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in self.model.state_dict().items() },
                'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in self.optimizer.state_dict().items() },
                'epoch': self.epoch,
                'losses': self.losses
        }

        torch.save(model_state, save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

        plt.plot(self.losses)
        plt.savefig(os.path.join(self.model_dir, 'loss.png'))
        plt.close()

    def restore(self, filename='weights'):
        try:
            link_name = os.path.join(self.model_dir, f'{filename}.pt')
            checkpoint = torch.load(link_name)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
            self.losses = checkpoint['losses']
            return True
        except FileNotFoundError:
            return False

    def train(self, device: torch.device):

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            self.config.batch_size,
            shuffle=True)

        max_steps = self.config.max_steps if self.config.max_steps else None

        with tqdm.tqdm(desc='Epoch', total=max_steps, leave=False,
                        position=0, initial=self.epoch) as epoch_pbar:

            while True:
                for batch_data, _ in tqdm.tqdm(dataloader, desc='Batch', leave=False, position=1):

                    if max_steps is not None and self.epoch > max_steps:
                        return

                    loss = self.train_step(batch_data, device)

                    if torch.isnan(loss).any():
                        raise RuntimeError(f'Detected NaN loss at epoch {self.epoch}.')

                    self.losses.append(loss.item())

                    if self.epoch % self.config.epoch_save_interval == 0:
                        self.save()

                epoch_pbar.update(1)
                self.epoch += 1

        self.save()

    def train_step(self, batch_data: torch.Tensor, device: torch.device):

        self.optimizer.zero_grad()

        # sample_data = _nested_map(sample_data, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        batch_data = batch_data.to(device)
        bacth_samples, _ = batch_data.shape
        self.noise_level = self.noise_level.to(device)

        t = torch.randint(0, len(self.model.config.noise_schedule), [bacth_samples], device=device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(batch_data)
        noisy_audio = noise_scale_sqrt * batch_data + (1.0 - noise_scale)**0.5 * noise

        predicted = self.model(noisy_audio, t)
        loss = self.loss_fn(noise, predicted.squeeze(1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return loss
