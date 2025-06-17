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

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from model_lib.learner import train, train_distributed
from model_lib.params import params


# Encontra uma porta livre para a comunicação entre os processos
def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  # Se houuver mais de uma CPU, divide o batch size entre as GPUs.
  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  
  #Se houver apenas uma GPU (ou nenhuma), ele executa o treinamento de forma normal com train
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir', # diretório para salvar checkpoints e logs.
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('data_dirs', nargs='+', # diretórios com arquivos .wav para o treinamento.
      help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps') # número máximo de passos de treinamento.
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training') # se setado, usa precisão de 16 bits.
  parser.add_argument('--n_check', '-n', default=1, type=int,
      help='number of save checkpoint') # número de checkpoints a salvar.
  main(parser.parse_args())
