import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

root = Path(__file__).parent
#load csv

def transform_categorical(df, stop_overfit = False):
    CATEGORICAL_COLUMNS = ['gender','ever_married','work_type','Residence_type','smoking_status']
    for original_column in CATEGORICAL_COLUMNS:
        df_to_add = pd.get_dummies(df[original_column])
        colnames = df_to_add.columns.values

        for i in colnames:
        if stop_overfit and i == colnames[-1]:
            break
        df[i] = df_to_add[i].tolist() #add new col

        df.drop(original_column) #remove original col

    return df


def get_csv(csv='participants_dataset.csv'):
    '''get csv from file an turn into a numpy array'''
    dt = pd.read_csv(str(root/csv))
    dt.drop(['ID'
    ]
    , axis=1, inplace=True)
    #remove rows without label
    return dt
dt = get_csv()
dt = dt.dropna()
dt = transform_categorical(dt)
dt = dt[dt['avg_glucose_level'] < 265]


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
