import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
		)
	def forward(self, x): return self.net(x)

class Down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.pool = nn.MaxPool2d(2)
		self.conv = ConvBlock(in_ch, out_ch)
	def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
		self.conv = ConvBlock(in_ch, out_ch)
	def forward(self, x1, x2):
		x1 = self.up(x1)
		dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
		x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
		return self.conv(torch.cat([x2, x1], dim=1))

class MaskOnlyObfuscator(nn.Module):
	def __init__(self, in_ch=3, base=32):
		super().__init__()
		self.inc = ConvBlock(in_ch, base)
		self.d1 = Down(base, base*2)
		self.d2 = Down(base*2, base*4)
		self.d3 = Down(base*4, base*8)
		self.u1 = Up(base*8 + base*4, base*4)
		self.u2 = Up(base*4 + base*2, base*2)
		self.u3 = Up(base*2 + base, base)
		self.out_mask = nn.Conv2d(base, 1, 1)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.d1(x1)
		x3 = self.d2(x2)
		x4 = self.d3(x3)
		y = self.u1(x4, x3)
		y = self.u2(y, x2)
		y = self.u3(y, x1)
		m = torch.sigmoid(self.out_mask(y))
		return m
