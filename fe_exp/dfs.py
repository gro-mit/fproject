#coding=utf-8
import sys
import numpy as np
import tensorflow as tf

#define para for the model
#LEARNING_RATE = 0.01
#BATCH_SIZE = 20
#N_EPOCHS = 10

class DeepFeaturnSelection(object):
    def __init__(self, learning_rate, batch_size, n_epochs):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.importances_ = None

    def fit(self, data, labels):
        labels = self.one_hot_encoding(labels)
        num, dim = data.shape
        
        X = tf.placeholder(tf.float32, [self.batch_size, dim], name = 'data')
        Y = tf.placeholder(tf.int32, [self.batch_size, 2], name = 'labels')
        
        #build network
        f_layer, feat_weights = self.feat_layer(X, dim)
        h0 = self.hidden_layer(f_layer, dim, 256, tf.nn.relu)
        h1 = self.hidden_layer(h0, 256, 128, tf.nn.relu)
        h2 = self.hidden_layer(h1, 128, 64, tf.nn.relu)
        Y_pred = self.hidden_layer(h2, 64, 2)

        #loss function
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred,\
                    labels=Y, name='loss')
        loss = tf.reduce_mean(entropy)

        optimizer = tf.train.GradientDescentOptimizer(self.lr).\
                    minimize(loss)

        #run session
        with tf.Session() as sess:
            data_batch, label_batch = self.get_batch(data, labels,\
                                        self.batch_size, self.epochs)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            n_batch = int(num/self.batch_size)
            i = 0
            total_loss = 0
            try:
                while not coord.should_stop():
                    X_batch, Y_batch = sess.run([data_batch, label_batch])
                    _, loss_batch = sess.run([optimizer, loss],\
                                    feed_dict={X: X_batch, Y:Y_batch})
                    total_loss += loss_batch
                    i += 1
                    if i%n_batch ==0:
                        print('avg loss epoch: {0}'.format(total_loss/n_batch))
                        total_loss = 0
            except tf.errors.OutOfRangeError:
                print('optimization done')
            finally:
                coord.request_stop()

            self.importances_ = np.array(feat_weights.eval())
            #top_indices = get_top_feat(50, feature_weights)
            #print(top_indices)
            #print(feature_weights[:,top_indices])


    def one_hot_encoding(self, labels):
        labels[labels < 0] = 0
        return tf.one_hot(labels, depth = 2)

    def feat_layer(self, inputs, n_in):
        feat_w = tf.Variable(tf.ones([1, n_in]), name = 'feat_weights')
        res = tf.multiply(inputs, feat_w)
        return res, feat_w

    def hidden_layer(self, inputs, n_in, n_out, activation = None):
        weights = tf.Variable(tf.truncated_normal([n_in, n_out], stddev = 0.1))
        biases = tf.Variable(tf.zeros([1, n_out])) + 0.1
        res = tf.matmul(inputs, weights) + biases

        if activation is None:
            return res
        else:
            return activation(res)

    def get_batch(self, data, labels, batch_size, n_epochs):
        input_queue = tf.train.slice_input_producer([data, labels], \
                    num_epochs = n_epochs, shuffle = True)
        data_batch, label_batch = tf.train.batch(input_queue, \
                        batch_size=batch_size, num_threads=1, capacity=64)
        return data_batch, label_batch

    def get_top_feat(self, n_feat, feat_weights):
        top_indices = np.argsort(-np.abs(feat_weights))[:, :n_feat].ravel()
        return top_indices


