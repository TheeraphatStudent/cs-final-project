import numpy as np
import tensorflow as tf

# Define the tensors directly with tf.Variable or tf.constant
t1 = tf.constant([[[1, 2], [2, 3]], [[4, 4], [5, 3]]])
t2 = tf.constant([[[7, 4], [8, 4]], [[2, 10], [15, 11]]])

# Concatenate along axis -2
d0 = tf.concat([t1, t2], axis=-2)

# Eager execution allows immediate evaluation
print(d0.numpy())
