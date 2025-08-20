from typing import Any, overload, Dict, Union, List, Sequence
import random

import torch
from torch.nn import functional as F
import numpy as np


class BatchTransform:

    @overload
    def __call__(self, batch: Any) -> Any: ...


class IdentityBatchTransform(BatchTransform):

    def __call__(self, batch: Any) -> Any:
        return batch
