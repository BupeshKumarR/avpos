import torch
import torch.nn.functional as F

class Losses:
	def __init__(self, lambda_util=5.0, lambda_id=1.0, lambda_adv=1.0, lambda_perc=0.0, lambda_reg=0.01, alpha=10.0,
	             lambda_tv: float = 0.0, lambda_focus: float = 0.0):
		self.lambda_util = lambda_util
		self.lambda_id = lambda_id
		self.lambda_adv = lambda_adv
		self.lambda_perc = lambda_perc
		self.lambda_reg = lambda_reg
		self.alpha = alpha
		self.perc_feat = None
		self.lambda_tv = lambda_tv
		self.lambda_focus = lambda_focus

	def util_loss(self, logits, y):
		return F.cross_entropy(logits, y)

	def identity_loss(self, fid_xp, fid_x, exp_variant=False, alpha=None):
		alpha = alpha if alpha is not None else self.alpha
		cos = F.cosine_similarity(fid_xp, fid_x, dim=1)
		if exp_variant:
			return torch.exp(alpha * cos).mean()
		return -cos.mean()

	def adv_loss(self, x_hat, x):
		return F.l1_loss(x_hat, x)

	def perceptual_loss(self, x_prime, x):
		if self.perc_feat is None:
			return x_prime.new_zeros(())
		return F.mse_loss(x_prime, x)

	def reg_loss(self, mask):
		return mask.mean()

	def tv_loss(self, mask):
		# total variation on mask to favor smooth regions
		dx = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
		dy = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
		return dx + dy

	def center_focus_loss(self, mask):
		# encourage higher mask in the center region of the image (rough face prior)
		b, c, h, w = mask.shape
		# build a soft 2D gaussian prior centered
		device = mask.device
		y = torch.linspace(-1, 1, steps=h, device=device).view(1, 1, h, 1)
		x = torch.linspace(-1, 1, steps=w, device=device).view(1, 1, 1, w)
		g = torch.exp(-(x**2 + y**2) / (2 * 0.3**2))  # sigma=0.3
		g = g / g.max()
		# want mask to be brighter where g is high => maximize (mask * g).mean() -> minimize negative
		return -(mask * g).mean()

	def total_for_O(self, logits_T_xp, y, fid_xp, fid_x, x_hat, x, mask, use_exp_id=False):
		l_util = self.util_loss(logits_T_xp, y)
		l_id = self.identity_loss(fid_xp, fid_x, exp_variant=use_exp_id)
		l_adv = self.adv_loss(x_hat, x)
		l_perc = self.perceptual_loss(x_hat, x)
		l_reg = self.reg_loss(mask)
		l_tv = self.tv_loss(mask) if self.lambda_tv > 0 else mask.new_zeros(())
		l_focus = self.center_focus_loss(mask) if self.lambda_focus > 0 else mask.new_zeros(())
		total = (self.lambda_util * l_util
		         - self.lambda_id * l_id
		         + self.lambda_adv * l_adv
		         + self.lambda_perc * l_perc
		         + self.lambda_reg * l_reg
		         + self.lambda_tv * l_tv
		         + self.lambda_focus * l_focus)
		return total, dict(util=l_util, id=l_id, adv=l_adv, perc=l_perc, reg=l_reg, tv=l_tv, focus=l_focus)

	def total_for_R(self, x_hat, x):
		return self.adv_loss(x_hat, x)
