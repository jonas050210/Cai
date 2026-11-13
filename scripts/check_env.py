#!/usr/bin/env python3
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
