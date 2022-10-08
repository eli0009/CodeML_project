from tabnanny import verbose
import tensorflow as tf
from main import model_path, features, labels

model = tf.keras.models.load_model(model_path)
acc = model.predict(
    features,
)

for a in acc:
    print(a * 100)


