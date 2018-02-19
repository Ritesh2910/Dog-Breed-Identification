import xml.etree.ElementTree
import dataset
from src.freezing import inception
from tf_record_utils import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pyprind
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from src.models import denseNN
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

images_root_dir = os.path.join(paths.STANFORD_DS_DIR, 'Images')
annotations_root_dir = os.path.join(paths.STANFORD_DS_DIR, 'Annotation')
CLASSES_COUNT = 120
INCEPTION_CLASSES_COUNT = 2048
INCEPTION_OUTPUT_FIELD = 'inception_output'
LABEL_ONE_HOT_FIELD = 'label_one_hot'
IMAGE_RAW_FIELD = 'image_raw'
INCEPTION_INPUT_TENSOR = 'DecodeJpeg/contents:0'
INCEPTION_OUTPUT_TENSOR = 'pool_3:0'
OUTPUT_NODE_NAME = 'output_node'
OUTPUT_TENSOR_NAME = OUTPUT_NODE_NAME + ':0'
HEAD_INPUT_NODE_NAME = 'x'
HEAD_INPUT_TENSOR_NAME = HEAD_INPUT_NODE_NAME + ':0'
DEV_SET_SIZE = 3000
TRAIN_SAMPLE_SIZE = 3000
CURRENT_MODEL_NAME = 'Dog_breed_Ritesh_Model_2'
HEAD_MODEL_LAYERS = [INCEPTION_CLASSES_COUNT, 1024, CLASSES_COUNT]

JPEG_EXT = '.jpg'
DATA_ROOT = 'C:/Users/rites/Desktop/dogbreed/dog-breeds-classification-master/data/'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
#TRAIN_TF_RECORDS = os.path.join(ROOT, 'dogs_train.tfrecords')
TRAIN_TF_RECORDS = os.path.join(DATA_ROOT, 'stanford.tfrecords')
TEST_TF_RECORDS = os.path.join(DATA_ROOT, 'dogs_test.tfrecords')
LABELS = os.path.join(DATA_ROOT, 'train', 'labels.csv')
BREEDS = 'C:/Users/rites/Desktop/dogbreed/dog-breeds-classification-master/data/breeds.csv'
IMAGENET_GRAPH_DEF = 'C:/Users/rites/frozen/inception/classify_image_graph_def.pb'
TEST_PREDICTIONS = 'predictions.csv'
METRICS_DIR = 'C:/Users/rites/metrics'
TRAIN_CONFUSION = os.path.join(METRICS_DIR, 'training_confusion.csv')
FROZEN_MODELS_DIR = 'C:/Users/rites/frozen'
CHECKPOINTS_DIR = 'C:/Users/rites/Desktop/dogbreed/dog-breeds-classification-master/src/training/checkpoints'
GRAPHS_DIR = 'graphs'
SUMMARY_DIR = 'summary'
STANFORD_DS_DIR = os.path.join(DATA_ROOT, 'stanford_ds')
STANFORD_DS_TF_RECORDS = os.path.join(DATA_ROOT, 'stanford.tfrecords')



def parse_annotation(path):
    xml_root = xml.etree.ElementTree.parse(path).getroot()
    object = xml_root.findall('object')[0]
    name = object.findall('name')[0].text.lower()
    bound_box = object.findall('bndbox')[0]

    return {
        'breed': name,
        'bndbox_xmin': bound_box.findall('xmin')[0].text,
        'bndbox_ymin': bound_box.findall('ymin')[0].text,
        'bndbox_xmax': bound_box.findall('xmax')[0].text,
        'bndbox_ymax': bound_box.findall('ymax')[0].text
    }


def parse_image(breed_dir, filename):
    path = os.path.join(images_root_dir, breed_dir, filename + '.jpg')
    print(path)
    img_raw = open(path, 'rb').read()

    return img_raw


def build_stanford_example(img_raw, inception_output, one_hot_label, annotation):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': bytes_feature(annotation['breed'].encode()),
        consts.IMAGE_RAW_FIELD: bytes_feature(img_raw),
        consts.LABEL_ONE_HOT_FIELD: float_feature(one_hot_label),
        consts.INCEPTION_OUTPUT_FIELD: float_feature(inception_output)}))

    return example



def get_int64_feature(example, name):
    return int(example.features.feature[name].int64_list.value[0])


def get_float_feature(example, name):
    return int(example.features.feature[name].float_list.value)


def get_bytes_feature(example, name):
    return example.features.feature[name].bytes_list.value[0]


def read_tf_record(record):
    features = tf.parse_single_example(
        record,
        features={
            consts.IMAGE_RAW_FIELD: tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            consts.LABEL_ONE_HOT_FIELD: tf.FixedLenFeature([consts.CLASSES_COUNT], tf.float32),
            consts.INCEPTION_OUTPUT_FIELD: tf.FixedLenFeature([2048], tf.float32)
        })
    return features


def read_test_tf_record(record):
    features = tf.parse_single_example(
        record,
        features={
            'id': tf.FixedLenFeature([], tf.string),
            consts.IMAGE_RAW_FIELD: tf.FixedLenFeature([], tf.string),
            consts.INCEPTION_OUTPUT_FIELD: tf.FixedLenFeature([2048], tf.float32)
        })
    return features


def features_dataset():
    filenames = tf.placeholder(tf.string)
    ds = tf.contrib.data.TFRecordDataset(filenames, compression_type='') \
        .map(read_tf_record)

    return ds, filenames


def test_features_dataset():
    filenames = tf.placeholder(tf.string)
    ds = tf.contrib.data.TFRecordDataset(filenames, compression_type='') \
        .map(read_test_tf_record)

    return ds, filenames


def one_hot_label_encoder():
    train_Y_orig = pd.read_csv(paths.BREEDS, dtype={'breed': np.str})
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_Y_orig['breed'])

    def encode(labels):
        return np.asarray(lb.transform(labels), dtype=np.float32)

    def decode(one_hots):
        return np.asarray(lb.inverse_transform(one_hots), dtype=np.str)

    return encode, decode

def _freeze_graph(graph_path, checkpoint_path, output_node_names, output_path):
    restore_op_name = 'save/restore_all'
    filename_tensor_name = 'save/Const:0'

    saved_path = freeze_graph.freeze_graph(
        graph_path, '', True, checkpoint_path,
        output_node_names, restore_op_name, filename_tensor_name,
        output_path, False, '', '')

    print('Frozen model saved to ' + output_path)

    return saved_path


def freeze_current_model(model_name, output_node_names):
    lines = open(os.path.join(paths.CHECKPOINTS_DIR, model_name + '_latest')).read().split('\n')
    last_checkpoint = [l.split(':')[2].replace('"', '').strip() for l in lines if 'model_checkpoint_path:' in l][0].split('\\')[2]
	
    checkpoint_path = os.path.join(paths.CHECKPOINTS_DIR, last_checkpoint)
    graph_path = os.path.join(paths.GRAPHS_DIR, model_name + '.pb')
    output_graph_path = os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb')

    #saver = tf.train.Saver()
    #checkpoint_path = saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=model_name)
    tf.train.write_graph(g, paths.GRAPHS_DIR, os.path.basename(graph_path), as_text=False)

    _freeze_graph(graph_path, checkpoint_path, output_node_names=output_node_names, output_path=output_graph_path)


def freeze_model(model_name, checkpoint, output_node_names):
    checkpoint_path = os.path.join(paths.CHECKPOINTS_DIR, checkpoint)
    graph_path = os.path.join(paths.GRAPHS_DIR, model_name + '.pbtext')
    output_graph_path = os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb')

    _freeze_graph(graph_path, checkpoint_path, output_node_names=output_node_names, output_path=output_graph_path)


def unfreeze_into_current_graph(model_path, tensor_names):
    with tf.gfile.FastGFile(name=model_path, mode='rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        g = tf.get_default_graph()

        tensors = {t: g.get_tensor_by_name(t) for t in tensor_names}

        return tensors


if __name__ == '__main__':
    one_hot_encoder, _ = dataset.one_hot_label_encoder()

    with tf.Graph().as_default(), \
         tf.Session().as_default() as sess, \
            tf.python_io.TFRecordWriter(paths.STANFORD_DS_TF_RECORDS,
                                        tf.python_io.TFRecordCompressionType.NONE) as writer:

        incept_model = inception.inception_model()


        def get_inception_ouput(img):
            inception_output = incept_model(sess, img).reshape(-1).tolist()
            return inception_output


        for breed_dir in [d for d in os.listdir(annotations_root_dir)]:
            print(breed_dir)
            for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
                print(annotation_file)
                annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))

                # print(annotation)

                one_hot_label = one_hot_encoder([annotation['breed']]).reshape(-1).tolist()
                image = parse_image(breed_dir, annotation_file)

                example = build_stanford_example(image, get_inception_ouput(image), one_hot_label, annotation)

                writer.write(example.SerializeToString())

        writer.flush()
        writer.close()

        print('Finished')

		

def convert(model_name, export_dir):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)

    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        tensors = freeze.unfreeze_into_current_graph(
            os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb'),
            tensor_names=[consts.INCEPTION_INPUT_TENSOR, consts.OUTPUT_TENSOR_NAME])

        raw_image_proto_info = tf.saved_model.utils.build_tensor_info(tensors[consts.INCEPTION_INPUT_TENSOR])
        probs_proto_info = tf.saved_model.utils.build_tensor_info(tensors[consts.OUTPUT_TENSOR_NAME])

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'image_raw': raw_image_proto_info},
                outputs={'probs': probs_proto_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                                             })

    builder.save()



	


def inception_model():
    tensors = freeze.unfreeze_into_current_graph(paths.IMAGENET_GRAPH_DEF,
                                                 tensor_names=[
                                                     consts.INCEPTION_INPUT_TENSOR,
                                                     consts.INCEPTION_OUTPUT_TENSOR])

    def forward(sess, image_raw):
        out = sess.run(tensors[consts.INCEPTION_OUTPUT_TENSOR], {tensors[consts.INCEPTION_INPUT_TENSOR]: image_raw})
        return out


    

    
    


def denseNNModel(input_node, layers, gamma=0.1):
    n_x = layers[0]
    n_y = layers[-1]
    L = len(layers)
    summaries = []
    Ws = []

    with tf.name_scope("placeholders"):
        # x = tf.placeholder(dtype=tf.float32, shape=(n_x, None), name="x")
        y = tf.placeholder(dtype=tf.float32, shape=(n_y, None), name="y")

    a = input_node

    with tf.name_scope("hidden_layers"):
        for l in range(1, len(layers) - 1):
            W = tf.Variable(np.random.randn(layers[l], layers[l - 1]) / tf.sqrt(layers[l - 1] * 1.0), dtype=tf.float32,
                            name="W" + str(l))
            Ws.append(W)
            summaries.append(tf.summary.histogram('W' + str(l), W))
            b = tf.Variable(np.zeros((layers[l], 1)), dtype=tf.float32, name="b" + str(l))
            summaries.append(tf.summary.histogram('b' + str(l), b))
            z = tf.matmul(W, a) + b
            a = tf.nn.relu(z)

    W = tf.Variable(np.random.randn(layers[L - 1], layers[L - 2]) / tf.sqrt(layers[L - 2] * 1.0), dtype=tf.float32,
                    name="W" + str(L - 1))
    summaries.append(tf.summary.histogram('W' + str(L - 1), W))
    b = tf.Variable(np.zeros((layers[L - 1], 1)), dtype=tf.float32, name="b" + str(L - 1))
    summaries.append(tf.summary.histogram('b' + str(L - 1), b))
    z = tf.matmul(W, a) + b

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y), logits=tf.transpose(z)))  # +\
        # gamma * tf.reduce_sum([tf.nn.l2_loss(w) for w in Ws])
        summaries.append(tf.summary.scalar('cost', cost))

    output = tf.nn.softmax(z, dim=0, name=consts.OUTPUT_NODE_NAME)

    return cost, output, y, summaries






def train_dev_split(sess, tf_records_path, dev_set_size=2000, batch_size=64, train_sample_size=4000):
    ds_, filename = dataset.features_dataset()

    ds = ds_.shuffle(buffer_size=20000)

    train_ds = ds.skip(dev_set_size).repeat()
    train_ds_iter = train_ds.shuffle(buffer_size=20000) \
        .batch(batch_size) \
        .make_initializable_iterator()

    train_sample_ds = ds.skip(dev_set_size)
    train_sample_ds_iter = train_sample_ds.shuffle(buffer_size=20000) \
        .take(train_sample_size) \
        .batch(train_sample_size) \
        .make_initializable_iterator()

    dev_ds_iter = ds.take(dev_set_size).batch(dev_set_size).make_initializable_iterator()

    sess.run(train_ds_iter.initializer, feed_dict={filename: tf_records_path})
    sess.run(dev_ds_iter.initializer, feed_dict={filename: tf_records_path})
    sess.run(train_sample_ds_iter.initializer, feed_dict={filename: tf_records_path})

    return train_ds_iter.get_next(), dev_ds_iter.get_next(), train_sample_ds_iter.get_next()


def error(x, output_probs, name):
    expected = tf.placeholder(tf.float32, shape=(consts.CLASSES_COUNT, None), name='expected')
    exp_vs_output = tf.equal(tf.argmax(output_probs, axis=0), tf.argmax(expected, axis=0))
    accuracy = 1. - tf.reduce_mean(tf.cast(exp_vs_output, dtype=tf.float32))
    summaries = [tf.summary.scalar(name, accuracy)]

    merged_summaries = tf.summary.merge(summaries)

    def run(sess, output, expected_):
        acc, summary_acc = sess.run([accuracy, merged_summaries],
                                    feed_dict={x: output, expected: expected_})

        return acc, summary_acc

    return run


def make_model_name(prefix, batch_size, learning_rate):
    return '%s_%d_%s' % (prefix, batch_size, str(learning_rate).replace('0.', ''))


def infer_test(model_name, output_probs, x):
    BATCH_SIZE = 20000

    with tf.Session().as_default() as sess:
        ds, filename = dataset.test_features_dataset()
        ds_iter = ds.batch(BATCH_SIZE).make_initializable_iterator()
        sess.run(ds_iter.initializer, feed_dict={filename: paths.TEST_TF_RECORDS})

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        lines = open(os.path.join(paths.CHECKPOINTS_DIR, model_name + '_latest')).read().split('\n')
        last_checkpoint = [l.split(':')[1].replace('"', '').strip() for l in lines if 'model_checkpoint_path:' in l][0]
        saver.restore(sess, os.path.join(paths.CHECKPOINTS_DIR, last_checkpoint))

        _, one_hot_decoder = dataset.one_hot_label_encoder()

        breeds = one_hot_decoder(np.identity(consts.CLASSES_COUNT))
        agg_test_df = None

        try:
            while True:
                test_batch = sess.run(ds_iter.get_next())

                inception_output = test_batch['inception_output']
                ids = test_batch['id']

                pred_probs = sess.run(output_probs, feed_dict={x: inception_output.T})

                #print(pred_probs.shape)

                test_df = pd.DataFrame(data=pred_probs.T, columns=breeds)
                test_df.index = ids

                if agg_test_df is None:
                    agg_test_df = test_df
                else:
                    agg_test_df = agg_test_df.append(test_df)

        except tf.errors.OutOfRangeError:
            print('End of the dataset')

        agg_test_df.to_csv(paths.TEST_PREDICTIONS, index_label='id', float_format='%.17f')

        print('predictions saved to %s' % paths.TEST_PREDICTIONS)
        
        

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
        

    
    
#Different modules are to be created in the python script with different seperate folders #

if __name__ == '__main__':
    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        next_train_batch, _, _ = train.train_dev_split(sess, paths.TRAIN_TF_RECORDS, batch_size=1)

        batch = sess.run(next_train_batch)
        img_raw = batch[consts.IMAGE_RAW_FIELD]
        img = adjust_brightness(tf.image.decode_jpeg(img_raw[0]), 0.1).eval()

        print(img.shape)

        plt.imshow(img)
        plt.show()

		
		
		


if __name__ == '__main__':
    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        ds, filenames = features_dataset()
        ds_iter = ds.shuffle(buffer_size=1000, seed=1).batch(10).make_initializable_iterator()
        next_record = ds_iter.get_next()

        sess.run(ds_iter.initializer, feed_dict={filenames: paths.TRAIN_TF_RECORDS})
        features = sess.run(next_record)

        _, one_hot_decoder = one_hot_label_encoder()

        print(one_hot_decoder(features['inception_output']))
        print(features['label'])
        print(features['inception_output'].shape)




if __name__ == '__main__':
    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        tensors = unfreeze_into_current_graph(paths.IMAGENET_GRAPH_DEF,
                                              tensor_names=[
                                                  consts.INCEPTION_INPUT_TENSOR,
                                                  consts.INCEPTION_OUTPUT_TENSOR])

        _, output_probs, y, _ = denseNN.denseNNModel(
            tf.reshape(tensors[consts.INCEPTION_OUTPUT_TENSOR], shape=(-1, 1), name=consts.HEAD_INPUT_NODE_NAME),
                consts.HEAD_MODEL_LAYERS,gamma=0.01)

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(paths.CHECKPOINTS_DIR, consts.CURRENT_MODEL_NAME))

        freeze_current_model(consts.CURRENT_MODEL_NAME, output_node_names=consts.OUTPUT_NODE_NAME)
		


if __name__ == '__main__':
    convert(consts.CURRENT_MODEL_NAME, export_dir='/tmp/dogs_1')
	
	
	



    return forward


if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        image_raw = tf.read_file('../../images/airedale.jpg').eval()

    g = tf.Graph()
    sess = tf.Session(graph=g)

    with g.as_default():
        model = inception_model()

    with g.as_default():
        out = model(sess, image_raw)
        print(out.shape)

		

if __name__ == '__main__':
    BATCH_SIZE = 64
    EPOCHS_COUNT = 15000
    LEARNING_RATE = 0.0001

    model_name = consts.CURRENT_MODEL_NAME

    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        next_train_batch, get_dev_ds, get_train_sample_ds = \
            train_dev_split(sess, paths.TRAIN_TF_RECORDS,
                            dev_set_size=consts.DEV_SET_SIZE,
                            batch_size=BATCH_SIZE,
                            train_sample_size=consts.TRAIN_SAMPLE_SIZE)

        dev_set = sess.run(get_dev_ds)
        dev_set_inception_output = dev_set[consts.INCEPTION_OUTPUT_FIELD]
        dev_set_y_one_hot = dev_set[consts.LABEL_ONE_HOT_FIELD]

        train_sample = sess.run(get_train_sample_ds)
        train_sample_inception_output = train_sample[consts.INCEPTION_OUTPUT_FIELD]
        train_sample_y_one_hot = train_sample[consts.LABEL_ONE_HOT_FIELD]

        x = tf.placeholder(dtype=tf.float32, shape=(consts.INCEPTION_CLASSES_COUNT, None), name="x")
        cost, output_probs, y, nn_summaries = denseNN.denseNNModel(
            x, consts.HEAD_MODEL_LAYERS, gamma=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        dev_error_eval = error(x, output_probs, name='test_error')
        train_error_eval = error(x, output_probs, name='train_error')

        nn_merged_summaries = tf.summary.merge(nn_summaries)
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, model_name))

        bar = pyprind.ProgBar(EPOCHS_COUNT, update_interval=1, width=60)

        saver = tf.train.Saver()

        for epoch in range(0, EPOCHS_COUNT):
            batch_features = sess.run(next_train_batch)
            batch_inception_output = batch_features[consts.INCEPTION_OUTPUT_FIELD]
            batch_y = batch_features[consts.LABEL_ONE_HOT_FIELD]

            _, summary = sess.run([optimizer, nn_merged_summaries], feed_dict={
                                      x: batch_inception_output.T,
                                      y: batch_y.T
                                  })

            writer.add_summary(summary, epoch)

            _, dev_summaries = dev_error_eval(sess, dev_set_inception_output.T, dev_set_y_one_hot.T)
            writer.add_summary(dev_summaries, epoch)

            _, train_sample_summaries = train_error_eval(sess, train_sample_inception_output.T, train_sample_y_one_hot.T)
            writer.add_summary(train_sample_summaries, epoch)

            writer.flush()

            if epoch % 10 == 0 or epoch == EPOCHS_COUNT:
                saver.save(sess, os.path.join(paths.CHECKPOINTS_DIR, model_name), latest_filename=model_name + '_latest')

            bar.update()

			
			
			
			
			
			


if __name__ == '__main__':
    with tf.Graph().as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(consts.INCEPTION_CLASSES_COUNT, None), name="x")
        _, output_probs, _, _ = denseNN.denseNNModel(
            x, consts.HEAD_MODEL_LAYERS, gamma=0.01)
        infer_test(consts.CURRENT_MODEL_NAME, output_probs, x)



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



