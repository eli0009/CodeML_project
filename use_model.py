import tensorflow as tf
from pathlib import Path
root = Path(__file__).parent
model = tf.keras.models.load_model(str(root / 'model'))