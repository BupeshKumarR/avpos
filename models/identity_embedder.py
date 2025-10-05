import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class IdentityEmbedder(nn.Module):
	def __init__(self, device='cpu'):
		super().__init__()
		self.backbone = InceptionResnetV1(pretrained='vggface2').eval()
		for p in self.backbone.parameters():
			p.requires_grad = False
		self.device = device

	def forward(self, x):
		if x.shape[-1] != 160 or x.shape[-2] != 160:
			x = torch.nn.functional.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
		xn = (x - 0.5) * 2.0
		emb = self.backbone(xn.to(self.device))
		emb = torch.nn.functional.normalize(emb, dim=1)
		return emb
