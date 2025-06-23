!pip install git+https://github.com/Marcel-Velez/audiocraft.git
!pip install pympler git+https://github.com/kkoutini/passt_hear21.git
!pip install git+https://github.com/Marcel-Velez/IE-in-AI-parler-workshop.git
!wget https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv



# general imports
import torch
import os
import locale
locale.getpreferredencoding = lambda: "UTF-8"
# import IPython.display as ipd
import numpy as np
import csv
from collections import defaultdict
import time
import copy
import typing
from torch import Tensor
from scipy.special import softmax
import pdb
import math
import pickle as pkl
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal, get_args
import random
import torch.nn as nn
from functools import partial
import wandb

# audio specific imports
import soundfile as sf
import torchaudio
from scipy.io import wavfile
import librosa

# music imports
from audiocraft.models import MusicGen
from audiocraft.models import musicgen
from audiocraft.modules.transformer import create_sin_embedding
from audiocraft.data.audio import audio_write
from hear21passt.base import load_model

from dotenv import load_dotenv
load_dotenv()

with open('./class_labels_indices.csv', "r") as file:
    csv_reader = csv.DictReader(file)
    labels = [row["display_name"] for row in csv_reader]


run = wandb.init(
    entity="IEinAI",
    project="ie_in_ai",
    name="generate_training_data"
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# loading the music model
music_model = MusicGen.get_pretrained("small")

def generate_music(model: nn.Module, prompts: str|list, duration: int=5, output_file: str, save_file: bool=True) -> torch.Tensor:
  if isinstance(prompts, str):
    prompts = [prompts]
  model.set_generation_params(duration=duration)  # generate 4 seconds.
  torch.manual_seed(42)
  generation = model.generate(prompts)

  # save the files if desired
  if isinstance(prompts, list) and len(prompts)>1:
    for ind, aud in enumerate(generation):
      if save_file:
        sf.write(f'{ind}_{output_file}', aud.cpu().numpy().squeeze(), MUSICGEN_SAMPLE_RATE)
  else:
      if save_file:
        sf.write(f'{output_file}', generation.cpu().numpy().squeeze(), MUSICGEN_SAMPLE_RATE)

  return generation