# Import all of the dependencies
import streamlit as st
from pathlib import Path
import tempfile
import subprocess
import imageio.v2 as imageio
import imageio_ffmpeg  # pip install imageio-ffmpeg

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model


# Set the layout to the streamlit app as centered
st.set_page_config(layout="centered")

st.title("LIP READING using 3D Convolutional Neural Networks")
st.text(
    "This application demonstrates lip reading using a 3D CNN model which analyzes lip motion from face video and outputs the predicted text without using audio."
)

# Resolve data directory relative to repository root (one level above this file)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "s1"

# Generating a list of options or videos
if DATA_DIR.exists():
    options = sorted([p.name for p in DATA_DIR.iterdir() if p.is_file() and p.suffix in {".mpg", ".mp4"}])
else:
    options = []

selected_video = st.selectbox("Choose video", options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    with col1:
        st.subheader("Input Video")
        file_path = str(DATA_DIR / selected_video)

        if Path(file_path).exists():
            # Create temporary output file
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_path = temp_output.name
            temp_output.close()

            # Use bundled ffmpeg binary from imageio-ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            try:
                with st.spinner("Converting video..."):
                    # H.264 conversion
                    cmd = [
                        ffmpeg_path,
                        "-i",
                        file_path,
                        "-c:v",
                        "libx264",
                        "-y",
                        output_path,
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0 or not Path(output_path).exists():
                    st.error("FFmpeg conversion failed; details below.")
                    st.code(result.stderr or "No stderr captured.")
                else:
                    with open(output_path, "rb") as video_file:
                        st.video(video_file.read())

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                try:
                    if Path(output_path).exists():
                        Path(output_path).unlink()
                except Exception:
                    pass
        else:
            st.error(f"Video file not found: {file_path}")

    with col2:
        st.subheader("Processed Frames")
        # Load tensors and prepare frames for GIF
        # load_data accepts either a python string or a tf.Tensor
        video, annotations = load_data(file_path)
        frames = tf.cast(video * 255, tf.uint8)  # scale to 0–255
        frames = tf.squeeze(frames, axis=-1).numpy()  # (T, H, W)

        # Write GIF and display; use a temp file to avoid writing into cwd
        gif_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        gif_path = gif_temp.name
        gif_temp.close()
        imageio.mimsave(gif_path, frames, fps=10)
        st.image(gif_path, width=800)
        try:
            Path(gif_path).unlink()
        except Exception:
            pass

        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        # Convert prediction to text
        st.subheader("Predicted Text")
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
        st.markdown(f"**{converted_prediction}**")
        