import logging
import sys
import warnings
import torch
import time

import numpy as np
import torch.nn.functional as f

from torch import Tensor
from torch.nn import Module
from typing import Union

packages = [logging, sys, warnings, torch, time, np, f]
classes = [Tensor, Module, Union]

# Types
NoneType = type(None)
BorderType = Union[int, float, NoneType]

# Constants
INF = 1e1
EPS = 1e-6

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration of the log
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )
