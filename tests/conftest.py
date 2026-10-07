import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(1)
