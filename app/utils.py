
import tensorflow as tf
from typing import List, Union
import cv2
import os
from pathlib import Path

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path: str) -> List[float]:
    """Read a video and return normalized grayscale frames.

    Args:
        path: Path to video file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and crop the mouth region.
        frame = tf.image.rgb_to_grayscale(frame)
        # NOTE: cropping indices are dataset-specific and assume GRID/s1 layout.
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {path}")

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Alignment file not found: {path}")

    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        raise ValueError(f"Could not read alignment file {path}: {e}")

    tokens = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2] != "sil":
            tokens = [*tokens, " ", parts[2]]

    if not tokens:
        raise ValueError(f"No valid tokens found in alignment file: {path}")

    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding="UTF-8"), (-1)))[1:]


def load_data(path: Union[str, tf.Tensor]):
    """Load frames and alignments for a given video path.

    The function accepts either a Python string path or a TensorFlow string tensor
    (which is what the Streamlit app currently passes). Paths to data and
    alignments are resolved relative to the repository root (one level above
    the `app/` package).
    """
    # Support being called with a tf.Tensor (from tf.convert_to_tensor(file_path)).
    if isinstance(path, tf.Tensor):
        path = path.numpy()
        if isinstance(path, (bytes, bytearray)):
            path = path.decode("utf-8")

    # Ensure we have a pure string now
    if not isinstance(path, str):
        raise TypeError("path must be a string or a tf.Tensor containing a string")

    file_name = os.path.splitext(os.path.basename(path))[0]

    repo_root = Path(__file__).resolve().parent.parent
    video_path = repo_root / "data" / "s1" / f"{file_name}.mpg"
    alignment_path = repo_root / "data" / "alignments" / "s1" / f"{file_name}.align"

    frames = load_video(str(video_path))
    alignments = load_alignments(str(alignment_path))

    return frames, alignments