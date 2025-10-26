
import tensorflow as tf
from typing import List
import cv2
import os 

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
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
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames could be read from video: {path}")
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    if not os.path.exists(path):
        raise FileNotFoundError(f"Alignment file not found: {path}")
    
    try:
        with open(path, 'r') as f: 
            lines = f.readlines() 
    except Exception as e:
        raise ValueError(f"Could not read alignment file {path}: {e}")
    
    tokens = []
    for line in lines:
        line = line.split()
        if len(line) >= 3 and line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    
    if not tokens:
        raise ValueError(f"No valid tokens found in alignment file: {path}")
    
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments