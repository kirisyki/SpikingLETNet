import torch.nn as nn
from quantization.int4_selfbuild import QLayer

def set_computing_mode(model:nn.Module, mode='pytorch'):
    for module in model.modules():
        if isinstance(module, QLayer):
            if mode=='pytorch':
                module.hardware_computing = False
            elif mode=='hardware':
                module.hardware_computing = True

def set_T_step(model:nn.Module, T=8):
    for module in model.modules():
        if isinstance(module, QLayer):
            module.T = T