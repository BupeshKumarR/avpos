import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
	"""Frozen early feature extractor (ResNet18 up to layer2)."""
	def __init__(self):
		super().__init__()
		from torchvision.models import resnet18
		backbone = resnet18(pretrained=True)
		self.stem = nn.Sequential(
			backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
		)
		self.layer1 = backbone.layer1
		self.layer2 = backbone.layer2
		for p in self.parameters():
			p.requires_grad = False

	def forward(self, x):
		x = self.stem(x)   # 1/4
		f1 = self.layer1(x)  # 1/4
		f2 = self.layer2(f1) # 1/8
		return f1, f2


class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
		)
	def forward(self, x):
		return self.net(x)


class Up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
		self.conv = ConvBlock(in_ch, out_ch)
	def forward(self, x, skip):
		x = self.up(x)
		# pad if needed
		dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
		x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
		return self.conv(torch.cat([skip, x], dim=1))


class FeatureSpaceObfuscator(nn.Module):
	"""
	Generates an obfuscated image x_obf by editing early feature maps and decoding.
	Also predicts a soft mask m (B,1,H,W) for budget control and blending.
	"""
	def __init__(self, in_ch: int = 3, base: int = 64):
		super().__init__()
		self.extractor = FeatureExtractor()
		# Project features and predict residual in feature space
		self.res_f2 = nn.Sequential(
			nn.Conv2d(128, base, 3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(base, 128, 3, padding=1)
		)
		self.res_f1 = nn.Sequential(
			nn.Conv2d(64, base//2, 3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(base//2, 64, 3, padding=1)
		)
		# Decoder to image space
		self.dec2 = ConvBlock(128, base)
		# up1 concatenates with f1 (64 channels): in_ch = base + 64
		self.up1 = Up(base + 64, base//2)
		# up0 concatenates with input x (3 channels): in_ch = base//2 + 3
		self.up0 = Up(base//2 + 3, base//4)
		self.to_img = nn.Conv2d(base//4, in_ch, 1)
		self.to_mask = nn.Conv2d(base//4, 1, 1)

	def forward(self, x: torch.Tensor):
		B, C, H, W = x.shape
		f1, f2 = self.extractor(x)   # f1: (B,64,H/4,W/4), f2: (B,128,H/8,W/8)
		# Add residuals in feature space (learned disruptions)
		f2_obf = f2 + self.res_f2(f2)
		f1_obf = f1 + self.res_f1(f1)
		# Decode
		z = self.dec2(f2_obf)
		z = self.up1(z, f1_obf)
		z = self.up0(z, x)  # align by padding inside Up
		# Predict image and mask
		img = torch.sigmoid(self.to_img(z))
		mask = torch.sigmoid(self.to_mask(z))
		# Resize mask to input size if needed
		if mask.size(2) != H or mask.size(3) != W:
			mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
		return img, mask


