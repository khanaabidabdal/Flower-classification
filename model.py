import torch
import torchvision

from torch import nn

def create_model(num_classes:int=5,
                 seed:int=42):
  weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
  transforms=weights.transforms()
  model=torchvision.models.efficientnet_b0(weights=weights)

  for param in model.parameters():
    param.requires_grad=False

  torch.manual_seed(42)
  model.classifier=nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=num_classes, # same number of output units as our number of classes
                    bias=True))

  return model,transforms
