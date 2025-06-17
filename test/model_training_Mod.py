# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
os.makedirs('logs', exist_ok=True)  # Cria a pasta 'logs' se ela não existir

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from model_lib.learner import train, train_distributed
from model_lib.params import params


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('data_dirs', nargs='+',
      help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  main(parser.parse_args())

import torch
from torch.utils.data import DataLoader, Dataset

# Classe do dataset (substitua com sua lógica de carregamento de dados)
class MyDataset(Dataset):
    def __init__(self, data_dirs):
        # Aqui você deve implementar o carregamento dos dados
        
        self.data = torch.randn(15000, 10)  # 100 amostras, 10 features cada
        self.labels = torch.randint(0, 2, (100000,))  # 100 labels (binário)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Inicialize o DataLoader com seu dataset
data_dirs = '/home/leticia.luz/Documents/trn_data'  
dataset = MyDataset(data_dirs)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Modelo simples para exemplo (substitua pelo seu modelo)
model = torch.nn.Sequential(
    torch.nn.Linear(10, 1),
    torch.nn.Sigmoid()
)

# Definição do otimizador e função de perda
criterion = torch.nn.BCELoss()  # Perda binária por exemplo
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Treinamento
losses = []
num_epochs = 300

for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# Salvar a loss em arquivo
output_dir = '/home/leticia.luz/Documents/generative_simulator/test/logs'
output_path = f'{output_dir}/loss_log.txt'


import os
os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    for epoch, loss in enumerate(losses):
        f.write(f'Epoch {epoch+1}, Loss: {loss:.4f}\n')
