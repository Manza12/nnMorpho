import logging
import sys
import warnings
import torch

import numpy as np
import torch.nn.functional as f

from torch import Tensor
from torch.nn import Module
from typing import Union

packages = [logging, sys, warnings, torch, np, f]
classes = [Tensor, Module, Union]

INF = 1e3
EPS = 1e-6

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )
