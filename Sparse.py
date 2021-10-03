import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as full

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name='w')
def sparse_selection(feature,shape):
    sparse_weight_ = weight_variable([shape,1])
    sparse_weight_ = full(tf.reshape(sparse_weight_,[-1,shape]), int(shape/2), activation_fn=tf.nn.relu)
    sparse_weight = tf.reshape(full(sparse_weight_, shape, activation_fn=None),[shape,1])
    selected_feature = tf.matmul(feature,sparse_weight)
    sparse_penalty = tf.norm(selected_feature,ord=1)
    return selected_feature, sparse_penalty, sparse_weight


