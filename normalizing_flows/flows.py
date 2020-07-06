import tensorflow as tf
from dataclasses import dataclass

from normalizing_flows.ops import expand_dims_n, dot_product


class FlowParameters:
    def transform(self, inputs): raise NotImplementedError()
    def jacobian_determinant(self, inputs): raise NotImplementedError()
    def jacobian_log_determinant(self, inputs):
        return tf.math.log(self.jacobian_determinant(inputs) + 1e-8)


class Flow(tf.Module):
    def parameterize(self, inputs) -> FlowParameters: raise NotImplementedError()


@dataclass
class PlanarFlowParameters(FlowParameters):
    # [parameter_batch_size, dims]
    w: tf.Tensor
    u: tf.Tensor
    b: tf.Tensor

    def transform(self, inputs):
        # input: [parameter_batch_size, ..., dims]
        # output: [parameter_batch-size, ..., dims]
        parameters_rank = tf.rank(self.w)
        expand_by = tf.rank(inputs) - parameters_rank
        expand_at = parameters_rank - 1
        broadcast_w = expand_dims_n(self.w, expand_at, expand_by)
        broadcast_u = expand_dims_n(self.u, expand_at, expand_by)
        broadcast_b = expand_dims_n(self.b, expand_at, expand_by)

        w_dot_inputs = dot_product(broadcast_w, inputs)
        x = tf.nn.tanh(w_dot_inputs + broadcast_b)
        y = broadcast_u * x
        return inputs + y

    def jacobian_determinant(self, inputs):
        # input: [parameter_batch_size, ..., dims]
        # output: [parameter_batch-size, ...]
        # Derivative of tanh(x) = 1 - tanh^2(x)
        parameters_rank = tf.rank(self.w)
        expand_by = tf.rank(inputs) - parameters_rank
        expand_at = parameters_rank - 1
        broadcast_w = expand_dims_n(self.w, expand_at, expand_by)
        broadcast_u = expand_dims_n(self.u, expand_at, expand_by)
        broadcast_b = expand_dims_n(self.b, expand_at, expand_by)

        w_dot_inputs = w_dot_inputs = dot_product(broadcast_w, inputs)
        v = tf.square(tf.nn.tanh(w_dot_inputs + broadcast_b))
        phi = 1. - v * broadcast_w
        u_dot_phi = dot_product(broadcast_u, phi)
        u_dot_phi = tf.squeeze(u_dot_phi, -1)
        return tf.abs(1. + u_dot_phi)


class PlanarFlow(Flow):
    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.hidden_layer = tf.keras.layers.Dense(400, activation=tf.nn.tanh)
        self.w_layer = tf.keras.layers.Dense(num_dims)
        self.u_layer = tf.keras.layers.Dense(num_dims)
        self.b_layer = tf.keras.layers.Dense(1)

    def parameterize(self, inputs):
        hidden = self.hidden_layer(inputs)
        w = self.w_layer(hidden)
        u = self.u_layer(hidden)
        b = self.b_layer(hidden)

        w_dot_u = tf.reduce_sum(w * u, axis=-1, keepdims=True)
        w_squared_norm = tf.reduce_sum(tf.square(w), axis=-1, keepdims=True) + 1e-8
        u_hat = u + (-1. + tf.nn.softplus(w_dot_u) - w_dot_u) * w / w_squared_norm

        return PlanarFlowParameters(w=w, u=u_hat, b=b)
