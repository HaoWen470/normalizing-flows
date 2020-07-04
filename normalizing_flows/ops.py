import tensorflow as tf


def expand_dims_n(x, axis, num_dims):
    if num_dims > 0:
        shape = tf.shape(x)
        new_shape = tf.concat([shape[:axis], [1] * num_dims, shape[axis:]], 0)
        return tf.reshape(x, new_shape)
    else:
        return x


def merge_dims(x, axis, num_dims):
    if num_dims > 0:
        shape = tf.shape(x)
        rank = tf.rank(x)
        if axis < 0:
            axis = axis + rank

        slice_size = tf.minimum(rank - axis, num_dims)
        if slice_size == 0:
            return x
        
        remaining_size = rank - slice_size - axis
        merged_size = tf.reduce_prod(tf.slice(shape, [axis], [slice_size]))
        new_shape = tf.concat([shape[:axis], [merged_size], tf.slice(shape, [axis+slice_size], [remaining_size])], 0)

        return tf.reshape(x, new_shape)
    else:
        return x


def dot_product(x, y):
    return tf.squeeze(tf.linalg.matmul(tf.expand_dims(x, -2), tf.expand_dims(y, -1)), -1)