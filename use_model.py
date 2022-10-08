import tensorflow as tf
from main import root, dt_features, dt_labels, preprocessed_inputs

model = tf.keras.models.load_model(str(root / 'model'))

dt_labels = tf.convert_to_tensor(dt_labels)
dt_features = preprocessed_inputs

# test_loss, test_acc = model.evaluate(dt_features, dt_labels, verbose=2)

# print('\nTest accuracy:', test_acc)

print(dt_features)