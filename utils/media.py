import os
import random
import torch

try:
	import torchvision
	from torchvision.io import read_image
	from torchvision.io import read_video
	_HAS_TV = True
except Exception:
	_HAS_TV = False
	read_image = None
	read_video = None

try:
	from PIL import Image
	import numpy as np
	_HAS_PIL = True
except Exception:
	_HAS_PIL = False
	Image = None
	np = None

try:
	import cv2
	_HAS_CV2 = True
except Exception:
	_HAS_CV2 = False
	cv2 = None

try:
	from decord import VideoReader, cpu
	_HAS_DECORD = True
except Exception:
	_HAS_DECORD = False
	VideoReader = None
	cpu = None

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
_VIDEO_EXTS = {'.avi', '.mp4', '.mov', '.mkv', '.webm', '.m4v'}

def is_image_file(path: str) -> bool:
	ext = os.path.splitext(path)[1].lower()
	return ext in _IMAGE_EXTS

def is_video_file(path: str) -> bool:
	ext = os.path.splitext(path)[1].lower()
	return ext in _VIDEO_EXTS

def load_image(path: str) -> torch.Tensor:
	# Returns float tensor [3,H,W] in [0,1]
	if _HAS_TV and read_image is not None:
		img = read_image(path)  # uint8 [C,H,W]
		if img.size(0) == 1:
			img = img.repeat(3, 1, 1)
		return img.float() / 255.0

	if _HAS_PIL:
		with Image.open(path) as im:
			im = im.convert("RGB")
			t = torch.from_numpy(np.array(im)).permute(2,0,1)  # type: ignore
			return t.float() / 255.0

	if _HAS_CV2:
		im = cv2.imread(path, cv2.IMREAD_COLOR)
		if im is None:
			raise RuntimeError(f"cv2 failed to read image: {path}")
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		t = torch.from_numpy(im).permute(2,0,1)
		return t.float() / 255.0

	raise RuntimeError("No available backend to read images. Install torchvision or pillow or opencv-python.")

def _video_to_tensor_with_tv(path: str, frame_index: int = None) -> torch.Tensor:
	vr, _, _ = read_video(path, output_format="THWC")  # [T,H,W,C] uint8
	if vr.numel() == 0:
		raise RuntimeError(f"Empty video: {path}")
	T = vr.shape[0]
	idx = frame_index if frame_index is not None else T // 2
	idx = max(0, min(T - 1, idx))
	frame = vr[idx]  # [H,W,C] uint8
	t = torch.as_tensor(frame).permute(2,0,1)  # [3,H,W]
	return t.float() / 255.0

def _video_to_tensor_with_decord(path: str, frame_index: int = None) -> torch.Tensor:
	vr = VideoReader(path, ctx=cpu(0))
	T = len(vr)
	if T == 0:
		raise RuntimeError(f"Empty video: {path}")
	idx = frame_index if frame_index is not None else T // 2
	idx = max(0, min(T - 1, idx))
	frame = vr[idx].asnumpy()  # [H,W,C] uint8
	t = torch.from_numpy(frame).permute(2,0,1)
	return t.float() / 255.0

def _video_to_tensor_with_cv2(path: str, frame_index: int = None) -> torch.Tensor:
	cap = cv2.VideoCapture(path)
	if not cap.isOpened():
		raise RuntimeError(f"cv2 cannot open video: {path}")
	T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if T <= 0:
		frames = []
		while True:
			ret, f = cap.read()
			if not ret:
				break
			frames.append(f)
		if not frames:
			cap.release()
			raise RuntimeError(f"No frames in video: {path}")
		T = len(frames)
		idx = frame_index if frame_index is not None else T // 2
		frame = frames[max(0, min(T-1, idx))]
	else:
		idx = frame_index if frame_index is not None else T // 2
		idx = max(0, min(T - 1, idx))
		cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
		ret, frame = cap.read()
		if not ret:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ret, frame = cap.read()
			if not ret:
				cap.release()
				raise RuntimeError(f"cv2 failed to read frame from: {path}")
	cap.release()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	t = torch.from_numpy(frame).permute(2,0,1)
	return t.float() / 255.0

def load_video_center_frame(path: str) -> torch.Tensor:
	if _HAS_DECORD:
		return _video_to_tensor_with_decord(path)
	if _HAS_TV and read_video is not None:
		return _video_to_tensor_with_tv(path)
	if _HAS_CV2:
		return _video_to_tensor_with_cv2(path)
	raise RuntimeError("No available backend to read videos. Install decord or torchvision (with ffmpeg) or opencv-python.")
