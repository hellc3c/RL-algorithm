import os
import random
import numpy as np
import torch


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch()