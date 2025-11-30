import os
import tensorflow as tf
import sys

# Ensure we can import from app directory
sys.path.append(os.getcwd())

from app.modelutil import load_model
from app.utils import load_data
from app.evaluation import evaluate_model

def mappable_function(path):
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

def main():
    print("Setting up data pipeline...")
    # Define data directory
    data_dir = os.path.join('data', 's1')
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please ensure you have the GRID corpus data in 'data/s1'")
        return

    # Create dataset pipeline (matching the notebook)
    # Note: We use a wildcard to match all .mpg files
    data = tf.data.Dataset.list_files(os.path.join(data_dir, '*.mpg'))
    data = data.shuffle(500, reshuffle_each_iteration=False)
    data = data.map(mappable_function)
    data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))
    data = data.prefetch(tf.data.AUTOTUNE)

    # Split data (matching the notebook: first 450 train, rest test)
    # train = data.take(450)
    test = data.skip(450)

    # Load model
    print("Loading model...")
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'checkpoint.weights.h5' exists in the root directory.")
        return
    
    # Evaluate
    print("Starting evaluation on test set... This might take a while.")
    try:
        wer, cer = evaluate_model(model, test)
        print("\n" + "="*30)
        print("EVALUATION RESULTS")
        print("="*30)
        print(f"Word Error Rate (WER):      {wer:.4f}")
        print(f"Character Error Rate (CER): {cer:.4f}")
        print("="*30)

        # Save results to file
        with open('evaluation_results.txt', 'w') as f:
            f.write("Evaluation Results\n")
            f.write("==================\n")
            f.write(f"Word Error Rate (WER):      {wer:.4f}\n")
            f.write(f"Character Error Rate (CER): {cer:.4f}\n")
        print(f"Results saved to 'evaluation_results.txt'")

    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
