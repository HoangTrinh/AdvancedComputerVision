import tensorflow as tf
import numpy as np


def knn(X_train, Y_train, X_test, k):
    neg_one = tf.constant(-1.0, dtype= tf.float64)
    # compute L2 distance
    distances = tf.reduce_sum(tf.square(tf.subtract(X_train, X_test)), axis=1)
    # find the furthest points * neg_one -> use top_k api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    values, indx = tf.nn.top_k(neg_distances, k)
    # get the correlative y
    top_y = tf.gather(Y_train, indx)
    return top_y

def get_label(top_y):
    counts = np.bincount(top_y.astype('int64'))
    return np.argmax(counts)
