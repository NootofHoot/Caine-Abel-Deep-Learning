import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

pipeline = pipeline(task="text-generation", model="EleutherAI/gpt-neo-1.3B", dtype=torch.float16, device=0)
