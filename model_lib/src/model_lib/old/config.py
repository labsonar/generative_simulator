import numpy as np

class Config():
    def __init__(self,
            batch_size=16,
            learning_rate=2e-4,
            max_grad_norm=None,
            n_check = 10,

            residual_layers=10,
            residual_channels=21,
            dilation_cycle_length=10,
            unconditional = True,
            noise_schedule=np.linspace(1e-4, 0.05, 10).tolist(),
            inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
            ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.n_check = n_check
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length
        self.unconditional = unconditional
        self.noise_schedule = noise_schedule
        self.inference_noise_schedule = inference_noise_schedule


params = AttrDict(
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    n_check = 10,

    residual_layers=10,
    residual_channels=21,
    dilation_cycle_length=10,
    unconditional = True,
    noise_schedule=np.linspace(1e-4, 0.05, 10).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
)
