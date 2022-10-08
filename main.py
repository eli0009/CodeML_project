import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

#laod csv
root = Path(__file__).parent
csv = 'participants_dataset.csv'
dt = pd.read_csv(str(root/csv))
dt.drop(['ID', 
'gender',
'ever_married',
'work_type',
'Residence_type',
'smoking_status'
]
, axis=1, inplace=True)
#remove rows without label
dt = dt.dropna()

features = dt.copy()
labels = features.pop('label')
feature = features

features = np.array(features)
model_path = str(root / 'model')

if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1),
    ])

    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                        optimizer = tf.keras.optimizers.Adam())
    model.fit(x = features, y = labels, epochs=10)
    model.save(model_path)