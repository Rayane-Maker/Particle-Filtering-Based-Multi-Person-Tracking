import torch
import torchvision
from torchvision.ops import nms

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


def read_choreo(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            print(len(parts))
    
file_path = "choreo.txt"
read_choreo(file_path)