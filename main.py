import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

root = Path(__file__).parent
#laod csv
def get_csv(csv='participants_dataset.csv'):
    '''get csv from file an turn into a numpy array'''
    dt = pd.read_csv(str(root/csv))
    dt.drop(['ID', 
    'work_type',
    ]
    , axis=1, inplace=True)
    #remove rows without label
    return dt
dt = get_csv()
dt = dt.dropna()
dt = dt[dt['avg_glucose_level'] < 5000]

features = dt.copy()
labels = features.pop('label')

features = np.array(features).astype('float32')
model_path = str(root / 'model')

if __name__ == "__main__":
    normalize = tf.keras.layers.Normalization()
    normalize.adapt(features)

    model = tf.keras.Sequential([
        normalize,
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1),
    ])

    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                        optimizer = tf.keras.optimizers.Adam())
    model.fit(x = features, y = labels, epochs=50)
    model.save(model_path)