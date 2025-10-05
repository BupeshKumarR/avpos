import torch
import torch.nn as nn
import torchvision.models as models

class UtilityTask(nn.Module):
	def __init__(self, num_classes=10, device='cpu'):
		super().__init__()
		base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
		for p in base.parameters():
			p.requires_grad = False
		in_feat = base.fc.in_features
		base.fc = nn.Linear(in_feat, num_classes)
		self.model = base.eval()
		self.device = device

	def forward(self, x):
		if x.shape[-1] != 224 or x.shape[-2] != 224:
			x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
		mean = x.new_tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
		std = x.new_tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
		x = (x - mean) / std
		return self.model(x.to(self.device))
