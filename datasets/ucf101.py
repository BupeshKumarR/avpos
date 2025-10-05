import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset

from utils.media import is_video_file, load_video_center_frame

class UCF101FramesDataset(Dataset):
	def __init__(self,
	             root: str,
	             split: str = 'train',
	             max_classes: Optional[int] = None,
	             max_per_class: Optional[int] = None,
	             resize: Tuple[int,int] = (224,224),
	             seed: int = 0):
		assert split in ('train','test','val')
		self.root = root
		self.resize = resize

		if not os.path.isdir(self.root):
			raise FileNotFoundError(f"UCF101 root not found: {self.root}")

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
			vids = [os.path.join(cdir, f) for f in os.listdir(cdir) if is_video_file(os.path.join(cdir, f))]
			vids.sort()
			if max_per_class is not None:
				vids = vids[:max_per_class]
			items.extend([(v, self.class_to_idx[c]) for v in vids])

		if len(items) == 0:
			raise RuntimeError(f"No video files found under {self.root}")

		self.items = items

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, idx: int):
		path, y = self.items[idx]
		frame = load_video_center_frame(path)
		if self.resize is not None:
			frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False).squeeze(0)
		return frame, torch.tensor(y, dtype=torch.long)
