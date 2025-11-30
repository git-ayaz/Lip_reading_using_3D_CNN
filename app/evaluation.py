import tensorflow as tf
import jiwer
from typing import List, Tuple
from app.utils import num_to_char

def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[float, float]:
    """
    Evaluates the model on the given dataset using WER and CER.

    Args:
        model: The trained Keras model.
        dataset: The validation/test dataset.

    Returns:
        A tuple containing (average_wer, average_cer).
    """
    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0

    for batch in dataset.as_numpy_iterator():
        videos, labels = batch
        # Predict
        yhat = model.predict(videos)
        # Decode
        input_length = [videos.shape[1]] * videos.shape[0]
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=input_length, greedy=True)[0][0].numpy()

        for i in range(len(videos)):
            # Convert ground truth to string
            original_text = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode('utf-8')
            
            # Convert prediction to string
            predicted_text = tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode('utf-8')

            # Calculate metrics
            # jiwer handles empty strings gracefully usually, but good to be safe
            if not original_text:
                continue
                
            wer = jiwer.wer(original_text, predicted_text)
            cer = jiwer.cer(original_text, predicted_text)

            total_wer += wer
            total_cer += cer
            num_samples += 1

    if num_samples == 0:
        return 0.0, 0.0

    avg_wer = total_wer / num_samples
    avg_cer = total_cer / num_samples
    
    return avg_wer, avg_cer
