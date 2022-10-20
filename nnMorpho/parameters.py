import logging
import sys
import warnings
import torch
import time

import numpy as np
import torch.nn.functional as f

import morphology_cuda
import greyscale_morphology_cpp
import binary_morphology_cpp
import cylindrical_binary_morphology_cpp
from torch import Tensor
from torch.nn import Module
from typing import Union, List, Any, Optional

# These lines are for avoid problems in PyCharm
packages = [logging, sys, warnings, torch, time, np, f]
packages_morphology = [morphology_cuda, greyscale_morphology_cpp, binary_morphology_cpp,
                       cylindrical_binary_morphology_cpp]
classes = [Tensor, Module, Union, List, Any, Optional]

# Types
NoneType = type(None)

# Constants
INF = torch.finfo(torch.float32).max
BLOCK_SHAPE = torch.tensor((32, 32, 1), dtype=torch.int16)

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration of the log
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )
