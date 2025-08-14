import typing
import math
import argparse

import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_func

class UnconditionalConfig():
    def __init__(self,
            residual_layers: int = 10,
            residual_channels: int = 21,
            dilation_cycle_length: int = 10,
            noise_schedule: np.array = np.linspace(1e-4, 0.05, 10).tolist(),
            inference_noise_schedule: typing.List[float] = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
            embedding_encoding_size: int = 128,
            embedding_size: int = 512,
            embedding_cycles: int = 4
            ):
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length
        self.noise_schedule = noise_schedule
        self.inference_noise_schedule = inference_noise_schedule
        self.embedding_encoding_size = embedding_encoding_size
        self.embedding_size = embedding_size
        self.embedding_cycles = embedding_cycles

    @staticmethod
    def add_arg_opt(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("UnconditionalConfig")
        group.add_argument('--residual_layers', type=int, default=10)
        group.add_argument('--residual_channels', type=int, default=21)
        group.add_argument('--dilation_cycle_length', type=int, default=10)
        group.add_argument('--noise_schedule', type=str, default='1e-4,0.05,10')
        group.add_argument('--inference_noise_schedule', type=str, default='0.0001,0.001,0.01,0.05,0.2,0.5')
        group.add_argument('--embedding_encoding_size', type=int, default=128)
        group.add_argument('--embedding_size', type=int, default=512)
        group.add_argument('--embedding_cycles', type=int, default=4)

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "UnconditionalConfig":
        # noise_schedule: format 'start,end,num_points'
        noise_schedule = list(map(float, args.noise_schedule.split(',')))
        if len(noise_schedule) == 3:
            noise_schedule = np.linspace(*noise_schedule[:-1], int(noise_schedule[-1])).tolist()
        else:
            noise_schedule = noise_schedule

        inference_noise_schedule = list(map(float, args.inference_noise_schedule.split(',')))

        return UnconditionalConfig(
            residual_layers=args.residual_layers,
            residual_channels=args.residual_channels,
            dilation_cycle_length=args.dilation_cycle_length,
            noise_schedule=noise_schedule,
            inference_noise_schedule=inference_noise_schedule,
            embedding_encoding_size=args.embedding_encoding_size,
            embedding_size=args.embedding_size,
            embedding_cycles=args.embedding_cycles
        )

def Conv1d(*args, **kwargs):
    layer = torch_nn.Conv1d(*args, **kwargs)
    torch_nn.init.kaiming_normal_(layer.weight)
    return layer

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

class DiffusionEmbedding(torch_nn.Module):
    def __init__(self, config: UnconditionalConfig):
        super().__init__()
        self.register_buffer('embedding',
                             self._build_embedding(len(config.noise_schedule),
                                                   config.embedding_size,
                                                   config.embedding_cycles),
                             persistent=False)
        self.projection1 = torch_nn.Linear(config.embedding_size, config.embedding_size)
        self.projection2 = torch_nn.Linear(config.embedding_size, config.embedding_size)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, diffusion_step):
        """ Interpolando o valor do passo t entre o passo anterior e posterior. """
        low_idx = torch.floor(diffusion_step).long()
        high_idx = torch.ceil(diffusion_step).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (diffusion_step - low_idx)

    def _build_embedding(self, n_diffusion_steps: int, encoding_size: int, cycles: int):
        # [T,1]                 -> [0, 1, ..., 50]
        steps = torch.arange(n_diffusion_steps).unsqueeze(1)

        # [1,encoding_size//2]  -> [0, 1, ..., 64]'
        dims = torch.arange(encoding_size//2).unsqueeze(0)

        # [T,encoding_size//2]  -> x = t * 10^((4 * n)/(63))
        table = steps * 10.0**(dims * cycles / (encoding_size//2-1.0))

        # [T,encoding_size]     -> [sin(x) cos(x)]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        return table


class ResidualBlock(torch_nn.Module):
    def __init__(self, index: int, config: UnconditionalConfig):
        super().__init__()
        dilation = 2**(index % config.dilation_cycle_length)
        self.diffusion_projection = torch_nn.Linear(config.embedding_size,
                                                    config.residual_channels)
        self.dilated_conv = Conv1d(config.residual_channels,
                                   2 * config.residual_channels,
                                   3,
                                   padding=dilation,
                                   dilation=dilation)
        self.output_projection = Conv1d(config.residual_channels,
                                        2 * config.residual_channels,
                                        1)

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        y = self.dilated_conv(y)

        chunk1, chunk2 = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(chunk1) * torch.tanh(chunk2)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffWave(torch_nn.Module):
    def __init__(self, config: UnconditionalConfig):
        super().__init__()
        self.config = config
        self.input_projection = Conv1d(1, config.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(config)

        self.residual_layers = torch_nn.ModuleList([
              ResidualBlock(i, config) for i in range(config.residual_layers)
        ])
        self.skip_projection = Conv1d(config.residual_channels, config.residual_channels, 1)
        self.output_projection = Conv1d(config.residual_channels, 1, 1)
        torch_nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = torch_func.relu(x)

        step_embedded = self.diffusion_embedding(diffusion_step)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, step_embedded)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = torch_func.relu(x)
        x = self.output_projection(x)
        return x

    def get_noise_level(self):
        beta = np.array(self.config.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        noise_level = torch.tensor(noise_level.astype(np.float32))
        return noise_level