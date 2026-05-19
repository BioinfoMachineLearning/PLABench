import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import AutoTokenizer, AutoModel, EsmModel
import yaml

# Import architectures
from .deepdta_arch import DeepDTA, Meta_regressor, label_smiles, label_sequence, CHARISOSMISET, CHARPROTSET, CHARISOSMILEN, CHARPROTLEN

# Import MEETA components (assuming they are defined in a separate file or I need to include them here?)
# The previous inference.py had MEETA definition inside or imported? 
# Previous inference.py had `from .model import MixingDTA`? No, it seemed to define classes or import them.
# Let's check what was in the previous inference.py.
# The previous file had:
# `class QKV(nn.Module)...`
# `class Regress(nn.Module)...`
# `class cross_AFT(nn.Module)...`
# `class DTA(nn.Module)...`
# `class MixingDTAPredictor...`

# To avoid losing MEETA code, I should have saved it or I need to re-include it.
# Ideally, I should put MEETA classes in `meeta_arch.py` as well.
# But for now I'll include MEETA classes inline or import if they exist.
# The user's file `MixingDTA/MEETA/model.py` exists. I can import from there or copy.
# Given I just created `deepdta_arch.py`, I should probably create `meeta_arch.py` from `MixingDTA/MEETA/model.py` + `MixingDTA/MEETA/main.py` parts?
# Or just copy the MEETA classes from the previous `inference.py` (which I can see in the `read_file` logs or just assume I need to fetch them).

# Actually, I can just use `MixingDTA/MEETA/model.py` directly if I add it to path?
# No, "avoid writing project code files to tmp...".
# I'll rely on `MixingDTA/MEETA/model.py` relative import if possible, but it's in a different project root.
# Better to define `meeta_arch.py`.

# For this step, I will focus on WARM START mainly, but I MUST NOT break cold start.
# So I will define `MEETA` classes here or in `meeta_arch.py`.
# Let's look at `MixingDTA/MEETA/model.py`.

# I'll create `meeta_arch.py` first, then `inference.py`.

pass
