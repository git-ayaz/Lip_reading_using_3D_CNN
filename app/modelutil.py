import os
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten


def load_model(weights_path: str = None) -> Sequential:
    """Builds the model and loads weights.

    Args:
        weights_path: Optional path to weights file. If not provided the function
            will look for `checkpoint.weights.h5` in the repository root.

    Returns:
        A Keras Sequential model with weights loaded if available.
    """
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer="he_normal", activation="softmax"))

    # Resolve weights path relative to repository root (one level above app/)
    if weights_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        weights_path = repo_root / "checkpoint.weights.h5"

    weights_path = Path(weights_path)
    if weights_path.exists():
        model.load_weights(str(weights_path))
    else:
        # Do not raise here; allow the caller to handle missing weights during dev.
        # Log a warning when running under a proper logging setup.
        pass

    return model