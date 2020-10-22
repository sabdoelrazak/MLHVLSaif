import torch
import numpy as np
from torch import FloatTensor, LongTensor
from typing import Tuple, List, Callable, Optional

input_dim = 16
hidden_dim = 32
output_dim = 15
x_to_h = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
h_to_h = torch.nn.Linear(hidden_dim, hidden_dim, True)
h_to_y = torch.nn.Linear(hidden_dim, output_dim, True)

x2 = np.asarray(x_to_h)

debug = True