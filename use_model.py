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

def get_label(display_graph=False):
    dt = get_csv('participants_dataset_predict.csv')
    dt.pop('label')
    dt = np.array(dt)
    
    acc = model.predict(
        dt
    )

    predictions = [] 
    values = []
    with open(str(root/'a.csv'), 'w') as fp:
        for a in acc:
            values.append(a[0] if a[0] < 0.5 else 0.5)
            prediction = True if a[0] >=0.35 else False
            print(int(prediction), file=fp)
            predictions.append(prediction)
    if display_graph:
        import matplotlib.pyplot as plt

        plt.plot(values, color='magenta', marker='o',mfc='pink' ) #plot the data
        plt.xticks(range(0,len(values)+1, 1)) #set the tick frequency on x-axis

        plt.ylabel('data') #set the label for y axis
        plt.xlabel('index') #set the label for x-axis
        plt.title("Plotting a list") #set the title of the graph
        plt.show() #display the graph
    print(len(dt))
    print(len(predictions))

if __name__ == '__main__':
    get_label()