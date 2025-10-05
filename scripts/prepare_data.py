import os
import tarfile
from pathlib import Path

ROOT = Path('/Users/bupesh/neu/2025/fall/avpos')
VGG_ROOT = ROOT / 'VGG-Face2'
DATA_DIR = VGG_ROOT / 'data'

TRAIN_DIR = VGG_ROOT / 'train'
TEST_DIR = VGG_ROOT / 'test'

TRAIN_TAR = DATA_DIR / 'vggface2_train.tar.gz'
TEST_TAR = DATA_DIR / 'vggface2_test.tar.gz'


def extract_tar_gz(src: Path, dst: Path):
	print(f"[prepare_data] Extracting {src} -> {dst}")
	dst.mkdir(parents=True, exist_ok=True)
	with tarfile.open(src, 'r:gz') as tar:
		# Safe extraction: ignore absolute paths, etc.
		def is_within_directory(directory, target):
			abs_directory = os.path.abspath(directory)
			abs_target = os.path.abspath(target)
			prefix = os.path.commonprefix([abs_directory, abs_target])
			return prefix == abs_directory

		def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
			for member in tar.getmembers():
				member_path = os.path.join(path, member.name)
				if not is_within_directory(path, member_path):
					raise Exception("Attempted Path Traversal in Tar File")
			tar.extractall(path, members=members, numeric_owner=numeric_owner)

		safe_extract(tar, path=str(dst))
	print(f"[prepare_data] Done: {src.name}")


def main():
	if TRAIN_DIR.is_dir():
		print(f"[prepare_data] Found VGGFace2 train dir: {TRAIN_DIR}")
	else:
		if TRAIN_TAR.is_file():
			extract_tar_gz(TRAIN_TAR, TRAIN_DIR)
		else:
			print(f"[prepare_data] Missing {TRAIN_TAR}. Skipping train extract.")

	if TEST_DIR.is_dir():
		print(f"[prepare_data] Found VGGFace2 test dir: {TEST_DIR}")
	else:
		if TEST_TAR.is_file():
			extract_tar_gz(TEST_TAR, TEST_DIR)
		else:
			print(f"[prepare_data] Missing {TEST_TAR}. Skipping test extract.")

	print("[prepare_data] UCF101 is a .rar archive; please extract to UCF101/<ClassName>/*.avi using unar/unrar.")

if __name__ == '__main__':
	main()
