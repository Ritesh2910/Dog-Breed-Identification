import os
import sys
import csv#
import urllib.request

import numpy as np
import pandas as pd
import tensorflow as tf

from src.common import consts
from src.data_preparation import dataset
from src.freezing import freeze
from src.common import paths


def infer(model_name, img_raw):
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        tensors = freeze.unfreeze_into_current_graph(
            os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb'),
            tensor_names=[consts.INCEPTION_INPUT_TENSOR, consts.OUTPUT_TENSOR_NAME])

        _, one_hot_decoder = dataset.one_hot_label_encoder()

        probs = sess.run(tensors[consts.OUTPUT_TENSOR_NAME],
                         feed_dict={tensors[consts.INCEPTION_INPUT_TENSOR]: img_raw})

        breeds = one_hot_decoder(np.identity(consts.CLASSES_COUNT)).reshape(-1)

        # print(breeds)

        df = pd.DataFrame(data={'prob': probs.reshape(-1), 'breed': breeds})


        return df.sort_values(['prob'], ascending=False)


def classify(resource_type, path):
    if resource_type == 'uri':
        response = urllib.request.urlopen(path)
        img_raw = response.read()
    else:
        with open(path, 'rb') as f:
            img_raw = f.read()

    return infer(consts.CURRENT_MODEL_NAME, img_raw)

if __name__ == '__main__':
    src = sys.argv[1]
    path = sys.argv[2] # uri to a dog image to classify
    with open(paths.BREEDS,'rt') as f:
        reader = csv.reader(f)
        breeds = [row[1] for row in reader]
        breeds[0] = 'id'
    outputcsv = []
    outputcsv.append(breeds)
    with open('csvoutput.csv', 'w', newline ='') as f:
        writer = csv.writer(f)
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                probsmatrix = classify(src, os.path.join(path,filename))
                probs = probsmatrix.sort_values(['breed']).iloc[:,1:].values.flatten().tolist()
                #np.insert(probs,0,filename.split('.')[0])
                probs.insert(0,filename.split('.')[0])
                #writer.writerow(probs)
                outputcsv.append(probs)
        writer.writerows(outputcsv)
	#probs = classify(src, path)

    #print(probs.sort_values(['prob'], ascending=False).take(range(120)))

