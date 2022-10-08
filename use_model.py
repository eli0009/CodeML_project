from tabnanny import verbose
import tensorflow as tf
from main import model_path, features, labels

model = tf.keras.models.load_model(model_path)
acc = model.predict(
    features,
)

success = 0
for a, b in zip(acc, labels):
    prediction = True if a[0] >=0 else False
    if prediction == b:
        success += 1
print(len(acc))
print(success)