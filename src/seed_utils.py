import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Фиксируем сиды для воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
