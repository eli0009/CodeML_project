import tensorflow as tf
import numpy as np
import pandas as pd
from main import (model_path, features, labels,
                  get_csv, root)

model = tf.keras.models.load_model(model_path)

def test_accuracy():
    acc = model.predict(
        features,
    )

    success = 0
    for a, b in zip(acc, labels):
        prediction = True if a[0] >=0 else False
        if prediction == b:
            success += 1
    print('accuracy: ' + str(success / len(acc) * 100))

def get_label():
    dt = get_csv('participants_dataset_predict.csv')
    dt.pop('label')
    dt = np.array(dt)
    
    acc = model.predict(
        dt
    )

    predictions = [] 
    with open(str(root/'a.csv'), 'w') as fp:
        for a in acc:
            prediction = True if a[0] >=0.5 else False
            print(int(prediction), file=fp)
            predictions.append(prediction)
    print(len(dt))
    print(len(predictions))

if __name__ == '__main__':
    get_label()