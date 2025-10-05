import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.block(x)

class Down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.pool = nn.MaxPool2d(2)
		self.conv = ConvBlock(in_ch, out_ch)

	def forward(self, x):
		return self.conv(self.pool(x))

class Up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
		self.conv = ConvBlock(in_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# Pad if needed
		diffY = x2.size(2) - x1.size(2)
		diffX = x2.size(3) - x1.size(3)
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class FiLM(nn.Module):
	def __init__(self, cond_dim, num_channels):
		super().__init__()
		self.gamma = nn.Linear(cond_dim, num_channels)
		self.beta = nn.Linear(cond_dim, num_channels)

	def forward(self, x, cond):
		g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
		b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
		return x * (1 + g) + b

class Obfuscator(nn.Module):
	def __init__(self, in_channels=3, base_ch=32, num_tasks=3, cond_dim=32):
		super().__init__()
		self.task_emb = nn.Embedding(num_tasks, cond_dim // 2)
		self.p_map = nn.Linear(1, cond_dim // 2)

		self.inc = ConvBlock(in_channels, base_ch)
		self.down1 = Down(base_ch, base_ch * 2)
		self.down2 = Down(base_ch * 2, base_ch * 4)
		self.down3 = Down(base_ch * 4, base_ch * 8)

		self.film = FiLM(cond_dim, base_ch * 8)

		self.up1 = Up(base_ch * 8 + base_ch * 4, base_ch * 4)
		self.up2 = Up(base_ch * 4 + base_ch * 2, base_ch * 2)
		self.up3 = Up(base_ch * 2 + base_ch, base_ch)
		self.out_img = nn.Conv2d(base_ch, in_channels, 1)
		self.out_mask = nn.Conv2d(base_ch, 1, 1)

	def forward(self, x, t_idx, p_scalar):
		t_vec = self.task_emb(t_idx)
		p_vec = self.p_map(p_scalar.view(-1, 1))
		cond = torch.cat([t_vec, p_vec], dim=1)

		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)

		x4 = self.film(x4, cond)
		y = self.up1(x4, x3)
		y = self.up2(y, x2)
		y = self.up3(y, x1)

		residual = self.out_img(y)
		# residual smoothing to reduce grainy artifacts
		residual = F.avg_pool2d(residual, kernel_size=3, stride=1, padding=1)
		mask_logits = self.out_mask(y)
		mask = torch.sigmoid(mask_logits)

		x_prime = torch.clamp(x * (1 - mask) + residual * mask, 0, 1)
		return x_prime, mask
