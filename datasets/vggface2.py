import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset

from utils.media import is_image_file, load_image

class VGGFace2Dataset(Dataset):
	def __init__(self,
	             root: str,
	             split: str = 'train',
	             max_classes: Optional[int] = None,
	             max_per_class: Optional[int] = None,
	             resize: Tuple[int,int] = (160,160),
	             seed: int = 0):
		assert split in ('train','test','val')
		self.root = os.path.join(root, split) if split != 'val' else os.path.join(root, 'test')
		# Handle nested archive layout like root/train/train/* created by some tarballs
		if os.path.isdir(os.path.join(self.root, split)):
			self.root = os.path.join(self.root, split)
		self.resize = resize

		if not os.path.isdir(self.root):
			raise FileNotFoundError(f"VGGFace2 split dir not found: {self.root}")

		classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
		if max_classes is not None:
			import random
			rng = random.Random(seed)
			rng.shuffle(classes)
			classes = classes[:max_classes]

		self.class_to_idx = {c:i for i,c in enumerate(classes)}
		items: List[Tuple[str,int]] = []
		for c in classes:
			cdir = os.path.join(self.root, c)
			paths = [os.path.join(cdir, f) for f in os.listdir(cdir) if is_image_file(os.path.join(cdir, f))]
			paths.sort()
			if max_per_class is not None:
				paths = paths[:max_per_class]
			items.extend([(p, self.class_to_idx[c]) for p in paths])

		if len(items) == 0:
			raise RuntimeError(f"No image files found under {self.root}")

		self.items = items

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, idx: int):
		path, y = self.items[idx]
		img = load_image(path)
		if self.resize is not None:
			img = torch.nn.functional.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False).squeeze(0)
		return img, torch.tensor(y, dtype=torch.long)
