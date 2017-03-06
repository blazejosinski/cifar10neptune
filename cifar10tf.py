import cPickle
import numpy as np
import tensorflow as tf
import sys
import itertools
from glob import glob
import random
import time
import Queue
from threading import Thread, Lock

# TODO - update data location, to where you have CIFAR-10 dataset.
DATA_DIR = "/mnt/ml-team/homes/blazej.osinski/cifar-10-batches-py/"

IMAGE_WIDTH = 32
S_IMAGE_WIDTH = 28

class DataFeeder(object):
    def __init__(self, filenames, download_in_background=False, default_shape=(S_IMAGE_WIDTH, S_IMAGE_WIDTH), default_batch_size=128):
        batches = []
        for f in filenames:
            dict = self.unpickle(f)
            data_scaled = dict["data"] / 255.0
            batches.append((data_scaled, dict["labels"]))#tf.cast(dict["labels"], tf.int32)))
        self.data = np.concatenate([d for (d,l) in batches])
        self.labels = list(itertools.chain.from_iterable([l for (d,l) in batches]))
        self.n = len(self.labels)
        self.index = 0
        self.index_lock = Lock()
        self.default_batch_size = default_batch_size
        self.default_shape = default_shape
        if download_in_background:
            self.data_queue = Queue.Queue(10)
            t = Thread(target=self.fill_queue)
            t.daemon = True
            t.start()

    def unpickle(self, file):
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def next_batch(self, batch_size=100):
        #return (self.data[:batch_size], self.labels[:batch_size])
        with self.index_lock:
            last_index = self.index + batch_size
            if last_index < self.n:
                res = (self.data[self.index:last_index], self.labels[self.index:last_index])
            else:
                last_index %= self.n
                res = (np.concatenate([self.data[self.index:], self.data[:last_index]]),
                       self.labels[self.index:] + self.labels[:last_index])
            self.index = last_index
            return res

    def next_batch_reshaped(self, shape, batch_size=100, repeat_image=1):
        images, labels = self.next_batch(batch_size)
        prev_shape = (3, IMAGE_WIDTH, IMAGE_WIDTH)
        def random_crop(a):
            b = a.reshape(prev_shape)
            x = random.randrange(prev_shape[1] - shape[0])
            y = random.randrange(prev_shape[2] - shape[1])
            if repeat_image == -1:
                x, y = 2,2
            # horizontal flip
            hrange = range(y, (y + shape[1])) if random.randint(0,1) == 1 else range((y + shape[1]-1), y-1, -1)
            return b[:, x:(x + shape[0]), hrange]
        if repeat_image>1:
            n_images = [np.array(list(itertools.imap(random_crop, images))) for r in xrange(0, repeat_image)]
        else:
            n_images = np.array(list(itertools.imap(random_crop, images)))
        return (n_images, labels)

    def cached_batch_reshaped(self):
        return self.data_queue.get()

    def fill_queue(self):
        while True:
            batch = self.next_batch_reshaped(self.default_shape, self.default_batch_size)
            self.data_queue.put(batch)

    def get_image_size(self):
        return self.data.shape[1]

    def get_n_examples(self):
        return self.n

def batch_norm(signal, phase_train, scope='batch_norm', decay=0.999, scale=False, shift=True):
    """
    Batch normalization
    Args:
       signal:      Tensor, 4D BHWD input maps (or any other arbitrary shape that make sense)
       phase_train: bool, true indicates training phase, false for test time (placeholder would be a good choice for this)
       scope:       string, variable scope
       decay:       float, exponential moving average decay rate
       scale:       bool, whether to allow output scaling
       shift:       bool, whether to allow adding offset
    Return:
       normed:      batch-normalized signal
    """
    return tf.contrib.layers.batch_norm(signal, is_training=phase_train, decay=decay, center=shift, scale=scale, scope=scope, trainable=True)

    train = tf.contrib.layers.batch_norm(signal, is_training=True, decay=decay, center=shift, scale=scale, scope=scope, trainable=True)
    test = tf.contrib.layers.batch_norm(signal, is_training=False, decay=decay, center=shift, scale=scale, reuse=True, scope=scope, trainable=True)
    return tf.cond(phase_train, lambda: train, lambda: test)


def weight_variable(shape, name=None):
    #return tf.get_variable(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name=None):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

reg = 3e-3

def regularization(Ws):
    loss = 0
    for w in Ws:
        loss = loss + tf.reduce_sum(w * w)
    return 0.5 * reg * loss

# * input - 4D tensor
# returns (layer, weight - for normalization purposes)
def conv_layer(input, filter_width, n_filters, apply_max_pool, name, batch_normalization_train=None):
    input_shape = input.get_shape().as_list()
    W = weight_variable([filter_width, filter_width, input_shape[-1], n_filters], name+"_weight")
    if batch_normalization_train == None:
        bias = bias_variable([n_filters], name+"_bias")
        inside = conv2d(input, W) + bias
    else:
        inside = tf.contrib.layers.batch_norm(conv2d(input, W), center=True, scale=False, is_training=True, scope=name+"_batch_norm")
    res = tf.nn.relu(inside)
    if apply_max_pool:
        res = max_pool_2x2(res)
    return (res, W)

def fc_layer(input, output_size, apply_relu, name):
    input_shape = input.get_shape().as_list()
    if len(input_shape) > 2:
        volume = reduce(lambda x, y: x * y, [d for d in input_shape[1:] if d and d > 0], 1)
        input = tf.reshape(input, [-1, volume])
    else:
        volume = input_shape[-1]

    W = weight_variable([volume, output_size], name+"_weight")
    bias = bias_variable([output_size], name+"_bias")

    res = tf.matmul(input, W) + bias
    if apply_relu:
        res = tf.nn.relu(res)
    return (res, W)

class Cifar_boxes():
    def __init__(self, args, ts):
        self.args = args
        self.ts = ts

    def run(self):
        print("Tensorflow version: ", tf.__version__)

        device = "/gpu:0"

        train_data_feeder = [DataFeeder(glob(DATA_DIR + "data_batch_*"), download_in_background=True, default_batch_size=512),
                             DataFeeder(glob(DATA_DIR + "data_batch_*"), download_in_background=True, default_batch_size=32)]
        test_data_feeder = DataFeeder(glob(DATA_DIR + "test_batch"))

        classes = 10
        starter_learning_rate = 0.001 #self.args.starting_learning_rate

        shallow = (self.args.shallow == "yes")

        use_batch_norm = (self.args.use_batch_norm == "yes") 

        x = tf.placeholder(tf.float32, [None, 3, S_IMAGE_WIDTH, S_IMAGE_WIDTH])
        y = tf.placeholder(tf.int64, [None])


        with tf.device(device):
            global_step = tf.Variable(0, trainable=False)

            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       10000, 0.9, staircase=True)
            train_phase = tf.placeholder(tf.bool, shape=[])
            if use_batch_norm:
                train_phase_p = train_phase
            else:
                train_phase_p = None 


            keep_prob = tf.placeholder(tf.float32, shape=[])

            W_list = []

            x_input = tf.transpose(x, [0, 2, 3, 1])

            (layer_1, w) = conv_layer(x_input, 3, 32, False, "layer_1", train_phase_p)
            W_list.append(w)

            if shallow:
                last_convoluted = layer_1
            else:
                (layer_2, w) = conv_layer(layer_1, 3, 64, True, "layer_2", train_phase_p)
                W_list.append(w)
             
                (layer_3, w) = conv_layer(layer_2, 3, 80, False, "layer_3", train_phase_p)
                W_list.append(w)
             
                (layer_4, w) = conv_layer(layer_3, 3, 96, True, "layer_4", train_phase_p)
                W_list.append(w)
             
                (layer_5, w) = conv_layer(layer_4, 3, 112, False, "layer_5", train_phase_p)
                W_list.append(w)

                (last_convoluted, w) = conv_layer(layer_5, 3, 128, True, "layer_6", train_phase_p)
                W_list.append(w)

            (fc_layer_1, w) = fc_layer(last_convoluted, 1024, True, "fc_layer_1")
            W_list.append(w)

            dropout_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob)

            (fc_layer_2, w) = fc_layer(dropout_layer_1, 512, True, "fc_layer_2")
            W_list.append(w)

            dropout_layer_2 = tf.nn.dropout(fc_layer_2, keep_prob)

            (output_layer, w) = fc_layer(dropout_layer_2, classes, False, "output_layer")
            W_list.append(w)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_layer, y))
            regularization_loss = regularization(W_list)
            full_loss = loss + regularization_loss
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(full_loss,
                                                                        global_step=global_step)

            y_pred = tf.nn.softmax(output_layer)
            correct_predictions = tf.equal(tf.argmax(y_pred, 1), y)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            def vote_correct_predictions(probs, labels):
                n = len(labels)
                arrays = np.split(probs, n)
                return [np.argmax(np.sum(a, axis=0)) == l for (a, l) in zip(arrays, labels)]

            def vote_correct_predictions_batched(probs_list, labels):
                probs = np.sum(probs_list, axis=0)
                pred = np.argmax(probs, axis=1)
                return np.equal(pred, labels)

            with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
                sess.run(tf.initialize_all_variables())

                shape = (S_IMAGE_WIDTH, S_IMAGE_WIDTH)

                val_batch_size = 100
                test_batches = [test_data_feeder.next_batch_reshaped(shape, val_batch_size, repeat_image=5) for i in
                                range(0,
                                      test_data_feeder.get_n_examples() / val_batch_size)]
                train_batches = [train_data_feeder[0].next_batch_reshaped(shape, val_batch_size) for i in
                                 range(0,
                                       train_data_feeder[0].get_n_examples() / val_batch_size)]
                losses = []
                times = []
                use_train_phase_bn = True

                generator_choice = 0
                for i in xrange(0, 1000000):
                    print "range ",i
                    ltime = time.time()
                    batch = train_data_feeder[generator_choice].cached_batch_reshaped()
                    print "first session is about to run!"
                    (_, loss_value) = sess.run((train_step, full_loss), feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8,
                                                    train_phase: True})
                    print "first session has run!"
                    times.append(time.time() - ltime)
                    losses.append(loss_value)
                    if i % 100 == 99:
                        self.ts.batch_processing_time.add(np.mean(times))
                        times = []
                        self.ts.loss.add(np.mean(losses))
                        losses = []
                        self.ts.learning_rate.add(sess.run(learning_rate))
                    if i % 1000 == 0:
                        test_correct_predictions =\
                            [vote_correct_predictions_batched([sess.run(y_pred, feed_dict={x: xx, keep_prob: 1, train_phase: use_train_phase_bn}) for xx in b[0]], b[1]) for b in test_batches]
                        tb = test_batches[0]
                        self.ts.generator_choice.add(generator_choice)
                        generator_choice = 0 if random.randint(0, 9) < 9 else 1
                        #print(test_correct_predictions)
                        #print(np.mean(list(itertools.chain.from_iterable(test_correct_predictions))))
                        self.ts.test_accuracy.add(
                            np.mean(list(itertools.chain.from_iterable(test_correct_predictions))))
                        self.ts.train_accuracy.add(np.mean(
                            [sess.run(accuracy, feed_dict={x: b[0], y: b[1], keep_prob: 1, train_phase: use_train_phase_bn}) for b in
                             train_batches]))
                    self.ts.learning_rate.add(sess.run(learning_rate))

def main():
    sys.exit(Cifar_boxes(None, None).run())

if __name__ == '__main__':
    main()
