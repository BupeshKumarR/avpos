import argparse
import os
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import trange
from torchvision.utils import save_image
import mlflow
import kornia
from facenet_pytorch import MTCNN
import random

from models.obfuscator import Obfuscator
from models.obfuscator_mask import MaskOnlyObfuscator
from models.reconstructor import Reconstructor
from models.identity_embedder import IdentityEmbedder
from models.utility_task import UtilityTask
from losses import Losses

from torch.utils.data import DataLoader, Dataset

class TinySyntheticImages(Dataset):
	def __init__(self, n=8, h=128, w=128, num_classes=10, seed=0):
		g = torch.Generator().manual_seed(seed)
		self.x = torch.rand((n, 3, h, w), generator=g)
		self.y = torch.randint(0, num_classes, (n,), generator=g)
	def __len__(self): return self.x.size(0)
	def __getitem__(self, idx): return self.x[idx], self.y[idx]

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic','vggface2','ucf101'])
	parser.add_argument('--data_root', type=str, default='')
	parser.add_argument('--split', type=str, default='train')
	parser.add_argument('--max_classes', type=int, default=5)
	parser.add_argument('--max_per_class', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=4)

	parser.add_argument('--lambda_id', type=float, default=1.0)
	parser.add_argument('--lambda_util', type=float, default=5.0)
	parser.add_argument('--lambda_adv', type=float, default=1.0)
	parser.add_argument('--lambda_perc', type=float, default=0.0)
	parser.add_argument('--lambda_reg', type=float, default=0.01)
	parser.add_argument('--lambda_tv', type=float, default=0.0)
	parser.add_argument('--lambda_focus', type=float, default=0.0)
	parser.add_argument('--use_exp_id', action='store_true')
	parser.add_argument('--num_tasks', type=int, default=3)
	parser.add_argument('--target_task', type=int, default=0)
	parser.add_argument('--privacy_budget', type=float, default=0.5)
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--steps', type=int, default=20)
	parser.add_argument('--log_every', type=int, default=5)
	parser.add_argument('--save_every', type=int, default=10)
	parser.add_argument('--save_dir', type=str, default='outputs')
	parser.add_argument('--lr_O', type=float, default=1e-4)
	parser.add_argument('--lr_R', type=float, default=1e-4)
	parser.add_argument('--mlflow_uri', type=str, default=str(Path.cwd() / 'mlruns'))
	parser.add_argument('--mlflow_experiment', type=str, default='mvp_obfuscator')
	parser.add_argument('--blur_sigma', type=float, default=8.0)
	parser.add_argument('--debug_threshold', type=float, default=0.0)
	parser.add_argument('--obfuscation_op', type=str, default='blur', choices=['blur','pixelate'])
	parser.add_argument('--pixel_block', type=int, default=24)
	parser.add_argument('--face_bbox_only', action='store_true')
	parser.add_argument('--mask_scale', type=float, default=1.0)
	parser.add_argument('--combined_op', action='store_true')
	parser.add_argument('--adaptive_pixelate', action='store_true')
	parser.add_argument('--min_block', type=int, default=4)
	parser.add_argument('--max_block', type=int, default=32)
	parser.add_argument('--quantize_levels', type=int, default=0)
	parser.add_argument('--noise_std', type=float, default=0.0)
	parser.add_argument('--shuffle_mode', type=str, default='none', choices=['none','patch','pixel'])
	parser.add_argument('--shuffle_patch', type=int, default=20)
	return parser.parse_args()

def build_loader(args):
	if args.dataset == 'synthetic':
		ds = TinySyntheticImages()
		return DataLoader(ds, batch_size=args.batch_size, shuffle=True)
	elif args.dataset == 'vggface2':
		assert args.data_root, "Provide --data_root to VGGFace2 root (containing train/test dirs)."
		from datasets.vggface2 import VGGFace2Dataset
		ds = VGGFace2Dataset(root=args.data_root, split=args.split, max_classes=args.max_classes, max_per_class=args.max_per_class, resize=(160,160))
		return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
	elif args.dataset == 'ucf101':
		assert args.data_root, "Provide --data_root to UCF101 root (class subdirs with videos)."
		from datasets.ucf101 import UCF101FramesDataset
		ds = UCF101FramesDataset(root=args.data_root, split=args.split, max_classes=args.max_classes, max_per_class=args.max_per_class, resize=(224,224))
		return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
	else:
		raise ValueError(args.dataset)

def apply_operator(img: torch.Tensor, face_bbox=None, args=None) -> torch.Tensor:
	"""Apply obfuscation operator to image crop"""
	H, W = img.shape[-2], img.shape[-1]
	
	if args.obfuscation_op == 'pixelate':
		if args.adaptive_pixelate and face_bbox is not None:
			# Use face bbox for adaptive block size
			x1, y1, x2, y2 = face_bbox[0]
			face_w, face_h = x2 - x1, y2 - y1
			face_size = max(face_w, face_h)
			blk = max(args.min_block, min(args.max_block, int(face_size / 20)))
		else:
			blk = max(2, int(args.pixel_block))
		
		img_small = torch.nn.functional.avg_pool2d(img, kernel_size=blk)
		pix = torch.nn.functional.interpolate(img_small, size=img.shape[-2:], mode='nearest')
		
		if args.combined_op:
			# Mix with blur 50/50 for robustness
			k_desired = max(3, int(2 * (3 * args.blur_sigma) + 1))
			k_max = max(3, min(int(H), int(W)))
			k = min(k_desired, k_max)
			k = k if (k % 2 == 1) else (k - 1)
			k = max(3, k)
			
			sigma_eff = float(args.blur_sigma)
			if k < k_desired:
				sigma_eff = max(0.5, (k - 1) / (2.0 * 3.0))
			
			gb = kornia.filters.GaussianBlur2d((k, k), (sigma_eff, sigma_eff))
			g = gb(img)
			return 0.5 * pix + 0.5 * g
		
		return pix
	
	# Default blur
	k_desired = max(3, int(2 * (3 * args.blur_sigma) + 1))
	k_max = max(3, min(int(H), int(W)))
	k = min(k_desired, k_max)
	k = k if (k % 2 == 1) else (k - 1)
	k = max(3, k)
	
	sigma_eff = float(args.blur_sigma)
	if k < k_desired:
		sigma_eff = max(0.5, (k - 1) / (2.0 * 3.0))
	
	gb = kornia.filters.GaussianBlur2d((k, k), (sigma_eff, sigma_eff))
	return gb(img)

def post_quantize_noise(t: torch.Tensor, args=None) -> torch.Tensor:
	"""Apply color quantization and noise"""
	q = t
	if int(args.quantize_levels) > 1:
		levels = int(args.quantize_levels)
		q = torch.round(q * (levels - 1)) / (levels - 1)
	if float(args.noise_std) > 0:
		q = torch.clamp(q + torch.randn_like(q) * float(args.noise_std), 0.0, 1.0)
	return q

def patch_shuffle(img: torch.Tensor, ps: int) -> torch.Tensor:
	"""Shuffle patches within image"""
	B, C, H, W = img.shape
	pad_h = (ps - (H % ps)) % ps
	pad_w = (ps - (W % ps)) % ps
	x = img
	if pad_h or pad_w:
		x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
	
	HB, WB = x.shape[-2], x.shape[-1]
	gh, gw = HB // ps, WB // ps
	
	# (B,C,gh,gw,ps,ps)
	patches = x.unfold(2, ps, ps).unfold(3, ps, ps).contiguous()
	patches = patches.view(B, C, gh*gw, ps, ps)
	idx = torch.randperm(gh*gw, device=img.device)
	patches = patches[:, :, idx, :, :]
	patches = patches.view(B, C, gh, gw, ps, ps)
	
	rows = []
	for i in range(gh):
		row = torch.cat([patches[:, :, i, j, :, :] for j in range(gw)], dim=3)
		rows.append(row)
	xrec = torch.cat(rows, dim=2)
	return xrec[:, :, :H, :W]

def pixel_permute(img: torch.Tensor, ps: int) -> torch.Tensor:
	"""Permute pixels within each patch"""
	B, C, H, W = img.shape
	pad_h = (ps - (H % ps)) % ps
	pad_w = (ps - (W % ps)) % ps
	x = img
	if pad_h or pad_w:
		x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
	
	HB, WB = x.shape[-2], x.shape[-1]
	gh, gw = HB // ps, WB // ps
	
	# (B,C,gh,gw,ps,ps)
	patches = x.unfold(2, ps, ps).unfold(3, ps, ps).contiguous()
	patches = patches.view(B, C, gh, gw, ps*ps)
	
	# Permute pixels within each patch independently
	perm = torch.randperm(ps*ps, device=img.device)
	patches = patches[:, :, :, :, perm]
	patches = patches.view(B, C, gh, gw, ps, ps)
	
	rows = []
	for i in range(gh):
		row = torch.cat([patches[:, :, i, j, :, :] for j in range(gw)], dim=3)
		rows.append(row)
	xrec = torch.cat(rows, dim=2)
	return xrec[:, :, :H, :W]

def obfuscate_full(img_bchw: torch.Tensor, O, args=None):
	"""Apply obfuscation to full image"""
	m = O(img_bchw)
	if args.debug_threshold > 0.0:
		m = (m > args.debug_threshold).float()
	
	# Optional mask scaling
	scale = max(0.0, float(args.mask_scale))
	m = torch.clamp(m * scale, 0.0, 1.0) if scale != 1.0 else m
	m3 = m.repeat(1, 3, 1, 1)
	
	op = apply_operator(img_bchw, args=args)
	
	# Apply shuffle transforms
	if args.shuffle_mode == 'patch':
		op = patch_shuffle(op, max(2, int(args.shuffle_patch)))
	elif args.shuffle_mode == 'pixel':
		op = pixel_permute(op, max(2, int(args.shuffle_patch)))
	
	op = post_quantize_noise(op, args=args)
	xp = torch.clamp((1 - m3) * img_bchw + m3 * op, 0, 1)
	return xp, m

def obfuscate_with_bbox(img_bchw: torch.Tensor, mtcnn_detector, O, args=None):
	"""Apply obfuscation only within detected face bounding boxes"""
	B, C, H, W = img_bchw.shape
	xp_list = []
	m_list = []
	
	for bi in range(B):
		img_i = img_bchw[bi:bi+1]
		
		# Convert to PIL-compatible for detection
		img_np = (img_i[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8')
		boxes, _ = mtcnn_detector.detect([img_np])
		
		if boxes is None or len(boxes[0]) == 0:
			# No faces detected: return original image and zero mask
			xp_list.append(img_i)
			m_list.append(torch.zeros((1, 1, H, W), device=img_bchw.device, dtype=img_bchw.dtype))
			continue
		
		# Select the largest face (first one)
		box = boxes[0][0].astype(int)
		x1, y1, x2, y2 = box
		
		# Clamp bbox to image dimensions
		x1 = max(0, min(W-1, x1))
		x2 = max(x1+1, min(W, x2))
		y1 = max(0, min(H-1, y1))
		y2 = max(y1+1, min(H, y2))
		
		# Clamp oversized boxes to at most 60% of frame on each axis
		bw = x2 - x1
		bh = y2 - y1
		max_w = int(0.6 * W)
		max_h = int(0.6 * H)
		
		if bw > max_w:
			cx = (x1 + x2) // 2
			hw = max_w // 2
			x1 = max(0, cx - hw)
			x2 = min(W, cx + hw)
		
		if bh > max_h:
			cy = (y1 + y2) // 2
			hh = max_h // 2
			y1 = max(0, cy - hh)
			y2 = min(H, cy + hh)
		
		# Ensure valid crop dimensions
		if x2 <= x1 or y2 <= y1:
			xp_list.append(img_i)
			m_list.append(torch.zeros((1, 1, H, W), device=img_bchw.device, dtype=img_bchw.dtype))
			continue
		
		# Extract face crop
		crop = img_i[:, :, y1:y2, x1:x2]
		
		# Generate mask for face crop only
		m_crop = O(crop)
		if args.debug_threshold > 0.0:
			m_crop = (m_crop > args.debug_threshold).float()
		
		# Optional mask scaling
		scale = max(0.0, float(args.mask_scale))
		m_crop = torch.clamp(m_crop * scale, 0.0, 1.0) if scale != 1.0 else m_crop
		m3_crop = m_crop.repeat(1, 3, 1, 1)
		
		# Apply obfuscation operator to face crop
		op_crop = apply_operator(crop, face_bbox=[(x1, y1, x2, y2)], args=args)
		
		# Apply shuffle transforms to the operator output
		if args.shuffle_mode == 'patch':
			op_crop = patch_shuffle(op_crop, max(2, int(args.shuffle_patch)))
		elif args.shuffle_mode == 'pixel':
			op_crop = pixel_permute(op_crop, max(2, int(args.shuffle_patch)))
		
		op_crop = post_quantize_noise(op_crop, args=args)
		
		# Blend obfuscated crop with original crop
		xp_crop = torch.clamp((1 - m3_crop) * crop + m3_crop * op_crop, 0, 1)
		
		# Paste back into original image
		xp_i = img_i.clone()
		xp_i[:, :, y1:y2, x1:x2] = xp_crop
		
		# Build full-size mask (zeros elsewhere)
		m_full = torch.zeros((1, 1, H, W), device=img_bchw.device, dtype=img_bchw.dtype)
		m_full[:, :, y1:y2, x1:x2] = m_crop
		
		xp_list.append(xp_i)
		m_list.append(m_full)
	
	# Concatenate results
	xp = torch.cat(xp_list, dim=0)
	m = torch.cat(m_list, dim=0)
	return xp, m

def main():
	args = parse_args()
	device = torch.device(args.device)

	# Prepare directories
	save_root = Path(args.save_dir)
	save_root.mkdir(parents=True, exist_ok=True)

	# MLflow setup
	mlflow.set_tracking_uri(args.mlflow_uri)
	mlflow.set_experiment(args.mlflow_experiment)

	# Initialize models
	O = MaskOnlyObfuscator().to(device)
	R = Reconstructor().to(device)
	Fid = IdentityEmbedder(device=device).to(device)
	T = UtilityTask(num_classes=101 if args.dataset == 'ucf101' else 1000, device=device).to(device)

	# Optional face detector for bbox-only application
	mtcnn = MTCNN(keep_all=False, device=device)

	# Freeze pretrained models
	for p in Fid.parameters():
		p.requires_grad = False
	for p in T.parameters():
		p.requires_grad = False

	# Initialize loss function
	crit = Losses(
		lambda_util=args.lambda_util,
		lambda_id=args.lambda_id,
		lambda_adv=args.lambda_adv,
		lambda_perc=args.lambda_perc,
		lambda_reg=args.lambda_reg,
		lambda_tv=args.lambda_tv,
		lambda_focus=args.lambda_focus
	)

	# Initialize optimizers
	opt_O = torch.optim.Adam([p for p in O.parameters() if p.requires_grad], lr=args.lr_O, betas=(0.5, 0.999))
	opt_R = torch.optim.Adam([p for p in R.parameters() if p.requires_grad], lr=args.lr_R, betas=(0.5, 0.999))

	# Build data loader
	loader = build_loader(args)
	it = iter(loader)

	with mlflow.start_run():
		# Log static parameters
		mlflow.log_params({
			'dataset': args.dataset,
			'data_root': args.data_root,
			'split': args.split,
			'max_classes': args.max_classes,
			'max_per_class': args.max_per_class,
			'lambda_id': args.lambda_id,
			'lambda_util': args.lambda_util,
			'lambda_adv': args.lambda_adv,
			'lambda_perc': args.lambda_perc,
			'lambda_reg': args.lambda_reg,
			'lambda_tv': args.lambda_tv,
			'lambda_focus': args.lambda_focus,
			'use_exp_id': args.use_exp_id,
			'privacy_budget': args.privacy_budget,
			'num_tasks': args.num_tasks,
			'lr_O': args.lr_O,
			'lr_R': args.lr_R,
			'blur_sigma': args.blur_sigma,
			'debug_threshold': args.debug_threshold,
			'obfuscation_op': args.obfuscation_op,
			'pixel_block': args.pixel_block,
			'face_bbox_only': args.face_bbox_only,
			'mask_scale': args.mask_scale,
			'combined_op': args.combined_op,
			'shuffle_mode': args.shuffle_mode,
			'shuffle_patch': args.shuffle_patch,
			'quantize_levels': args.quantize_levels,
			'noise_std': args.noise_std,
			'adaptive_pixelate': args.adaptive_pixelate,
			'min_block': args.min_block,
			'max_block': args.max_block,
		})

		for step in trange(args.steps):
			try:
				x, y = next(it)
			except StopIteration:
				it = iter(loader)
				x, y = next(it)
			
			x = x.to(device)
			y = y.to(device)

			B = x.size(0)
			t_idx = torch.full((B,), args.target_task, device=device, dtype=torch.long)
			p_scalar = torch.full((B,), args.privacy_budget, device=device)

			# Apply obfuscation
			if args.face_bbox_only:
				x_prime, mask = obfuscate_with_bbox(x, mtcnn, O, args)
			else:
				x_prime, mask = obfuscate_full(x, O, args)
			
			# Forward pass through networks
			with torch.no_grad():
				logits_T_x = T(x)
				fid_x = Fid(x)
			
			logits_T_xp = T(x_prime)
			fid_xp = Fid(x_prime)
			x_hat = R(x_prime)

			# Update obfuscator
			opt_O.zero_grad(set_to_none=True)
			l_total_O, parts = crit.total_for_O(logits_T_xp, y, fid_xp, fid_x, x_hat.detach(), x, mask, use_exp_id=args.use_exp_id)
			l_total_O.backward()
			opt_O.step()

			# Update reconstructor
			x_prime_detached = x_prime.detach()
			x_hat_R = R(x_prime_detached)
			opt_R.zero_grad(set_to_none=True)
			l_R = crit.total_for_R(x_hat_R, x)
			l_R.backward()
			opt_R.step()

			# Periodic metrics logging & prints
			if step % args.log_every == 0:
				with torch.no_grad():
					cos_sim = torch.nn.functional.cosine_similarity(fid_xp, fid_x, dim=1).mean().item()
				
				mask_min = mask.min().item()
				mask_mean = mask.mean().item()
				mask_max = mask.max().item()
				
				print(f"step {step} | L_O={l_total_O.item():.3f} (util={parts['util'].item():.3f} id={parts['id'].item():.3f} adv={parts['adv'].item():.3f} reg={parts['reg'].item():.3f}) | L_R={l_R.item():.3f} | cos={cos_sim:.3f} | mask[min/mean/max]={mask_min:.3f}/{mask_mean:.3f}/{mask_max:.3f}")
				
				mlflow.log_metrics({
					'L_O': float(l_total_O.item()),
					'L_R': float(l_R.item()),
					'L_util': float(parts['util'].item()),
					'L_id': float(parts['id'].item()),
					'L_adv': float(parts['adv'].item()),
					'L_reg': float(parts['reg'].item()),
					'cos_sim': float(cos_sim),
					'mask_min': float(mask_min),
					'mask_mean': float(mask_mean),
					'mask_max': float(mask_max),
				}, step=step)

			# Periodic artifact saving
			if step % args.save_every == 0:
				with torch.no_grad():
					# Pick first sample in batch
					x_0 = x[0:1].cpu()
					xp_0 = x_prime[0:1].cpu()
					m_0 = mask[0:1].cpu()
					
					# Tile for visibility
					mask_rgb = m_0.repeat(1, 3, 1, 1)
					grid_path = save_root / f'step_{step:06d}.png'
					save_image(torch.cat([x_0, xp_0, mask_rgb], dim=0), str(grid_path), nrow=3)
					mlflow.log_artifact(str(grid_path))
					
					# Save separate panels for clarity
					save_image(x_0, str(save_root / f'step_{step:06d}_orig.png'))
					save_image(xp_0, str(save_root / f'step_{step:06d}_blended.png'))
					save_image(mask_rgb, str(save_root / f'step_{step:06d}_mask.png'))

if __name__ == '__main__':
	main()