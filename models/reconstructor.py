import torch
import torch.nn as nn

class ReconBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.net(x)

class Reconstructor(nn.Module):
	def __init__(self, in_ch=3, base=32):
		super().__init__()
		self.e1 = ReconBlock(in_ch, base)
		self.d1 = ReconBlock(base, base)
		self.out = nn.Conv2d(base, in_ch, 1)

	def forward(self, x_prime):
		h = self.e1(x_prime)
		h = self.d1(h)
		return torch.clamp(self.out(h), 0, 1)
