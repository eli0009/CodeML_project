import tensorflow as tf
import numpy as np
import pandas as pd 
from pathlib import Path

#laod csv
root = Path(__file__).parent
csv = 'participants_dataset.csv'
dt = pd.read_csv(str(root/csv))
dt.drop(['ID'], axis=1, inplace=True)
#remove rows without label
dt = dt.dropna()

#get features and label
dt_features = dt.copy()
dt_labels = dt_features.pop('label')

#set symbolic link
inputs = {}

for name, column in dt_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

# processing
preprocessed_inputs = []
# process non string
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = tf.keras.layers.Normalization()
norm.adapt(np.array(dt[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs.append(all_numeric_inputs)

# process string
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(dt_features[name]))
    one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

# concat processed input
preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)
dt_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

dt_features_dict = {name: np.array(value)
                    for name, value in dt_features.items()}
# features_dict = {name:values[:1] for name, values in dt_features_dict.items()}
# dt_preprocessing(features_dict)

def train_model(name='model'):

    def dt_model(preprocessing_head, inputs):
        body = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(1)
        ])

        preprocessed_inputs = preprocessing_head(inputs)
        result = body(preprocessed_inputs)
        model = tf.keras.Model(inputs, result)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        optimizer=tf.keras.optimizers.Adam())
        return model

    dt_model = dt_model(dt_preprocessing, inputs)

    dt_model.fit(x=dt_features_dict, y=dt_labels, epochs=10)

    dt_model.save(str(root / 'model'))

if __name__ == "__main__":

    def print_data():
        '''Print the data'''
        print(dt_labels.head())
        print(dt_features.head())
    
    def print_model(image='model'):
        '''print the model as image, image is a filename as a string'''
        image = image + '.png'
        tf.keras.utils.plot_model(model = dt_preprocessing, rankdir="LR", dpi=72, show_shapes=True, to_file=str(root/image))

    def print_feature():
        print(dt_features_dict)
        print(features_dict)
    
    def print_tensors():
        for i in range(1, len(dt_features_dict)):
            features_dict = {name:values[i-1:i] for name, values in dt_features_dict.items()}
            print(dt_preprocessing(features_dict))


    # print_data()
    # print(preprocessed_inputs)
    print_model('model')
    # print_tensors()