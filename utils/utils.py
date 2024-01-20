import numpy as np
import torch

def set_seed(seed: int):
    """
    Set the seed for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    