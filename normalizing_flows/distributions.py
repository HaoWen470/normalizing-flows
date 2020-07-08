from dataclasses import dataclass

from typing import Tuple, Sequence, NamedTuple, Any, List
from math import pi

import tensorflow as tf

from normalizing_flows.flows import FlowParameters, Flow
from normalizing_flows.ops import expand_dims_n


class DistributionParameters:
    @property
    def parameters_batch_rank(self) -> tf.Tensor: raise NotImplementedError()
    def log_likelihood(self, inputs) -> tf.Tensor: raise NotImplementedError()
    def sample_batch(self, batch_size: int = 1) -> tf.Tensor: raise NotImplementedError()
    def sample(self) -> tf.Tensor: 
        batch = self.sample_batch(batch_size=1)
        return tf.squeeze(batch, axis=int(self.parameters_batch_rank))


class ParameterizableDistribution(tf.Module):
    def parameterize(self, inputs) -> DistributionParameters: raise NotImplementedError()


@dataclass
class DiagonalGaussianParameters(DistributionParameters):
    mean: tf.Tensor
    diagonal_covariance: tf.Tensor

    @property
    def parameters_batch_rank(self):
        return tf.rank(self.mean) - 1

    def log_likelihood(self, inputs):
        shape = tf.shape(self.mean)

        parameters_rank = tf.rank(self.mean)
        expand_by = tf.rank(inputs) - parameters_rank
        expand_at = parameters_rank - 1
        broadcast_mean = expand_dims_n(self.mean, expand_at, expand_by)
        broadcast_diagonal_covariance = expand_dims_n(self.diagonal_covariance, expand_at, expand_by)

        k = tf.cast(shape[-1], dtype=tf.float32)
        shifted_inputs = inputs - broadcast_mean

        constant_term = -0.5 * k * tf.math.log(2. * pi)
        determinant_term = -0.5 * tf.reduce_sum(tf.math.log(broadcast_diagonal_covariance), -1)
        exp_term = -0.5 * tf.reduce_sum(tf.square(shifted_inputs) * (1. / broadcast_diagonal_covariance), -1)

        return constant_term + determinant_term + exp_term

    def sample_batch(self, batch_size: int = 1) -> tf.Tensor:
        shape = tf.shape(self.mean)

        parameters_batch_rank = self.parameters_batch_rank
        multiples = tf.ones_like(shape, dtype=tf.int32)
        multiples = tf.concat([multiples[:parameters_batch_rank], [batch_size], multiples[parameters_batch_rank:]], 0)

        mean = tf.tile(tf.expand_dims(self.mean, parameters_batch_rank), multiples)
        stddev = tf.tile(tf.expand_dims(tf.sqrt(self.diagonal_covariance), parameters_batch_rank), multiples)
        
        output_shape = tf.concat([shape[:parameters_batch_rank], [batch_size], shape[parameters_batch_rank:]], 0)

        return tf.random.normal(output_shape) * stddev + mean


class DiagonalGaussianLayer(ParameterizableDistribution):
    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.hidden_layer = tf.keras.layers.Dense(num_dims, activation=tf.nn.tanh)
        self.mean_layer = tf.keras.layers.Dense(num_dims)
        self.diagonal_covariance_layer = tf.keras.layers.Dense(num_dims, activation=tf.exp)

    def parameterize(self, inputs) -> DiagonalGaussianParameters:
        hidden = self.hidden_layer(inputs)
        mean = self.mean_layer(hidden)
        diagonal_covariance = self.diagonal_covariance_layer(hidden)
        return DiagonalGaussianParameters(mean=mean, diagonal_covariance=diagonal_covariance)


class StandardGaussianLayer(ParameterizableDistribution):
    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims

    def parameterize(self, parameter_batch_shape) -> DiagonalGaussianParameters:
        outputs_shape = tf.concat([parameter_batch_shape, [self.num_dims]], 0)

        mean = tf.zeros(outputs_shape, dtype=tf.float32)
        diagonal_covariance = tf.ones(outputs_shape, dtype=tf.float32)

        return DiagonalGaussianParameters(mean=mean, diagonal_covariance=diagonal_covariance)


@dataclass
class IndependentBernoulliParameters(DistributionParameters):
    p: tf.Tensor

    @property
    def parameters_batch_rank(self):
        return tf.rank(self.p) - 1

    def log_likelihood(self, inputs):
        shape = tf.shape(self.p)

        parameters_rank = tf.rank(self.p)
        expand_by = tf.rank(inputs) - parameters_rank
        expand_at = parameters_rank - 1
        broadcast_p = expand_dims_n(self.p, expand_at, expand_by)

        log_p = tf.math.log(broadcast_p + 1e-8)
        log_1mp = tf.math.log(1. - broadcast_p + 1e-8)

        independent_ll = inputs * log_p + (1. - inputs) * log_1mp
        return tf.reduce_sum(independent_ll, -1)

    def sample_batch(self, batch_size: int = 1) -> tf.Tensor:
        shape = tf.shape(self.p)

        parameters_batch_rank = self.parameters_batch_rank
        multiples = tf.ones_like(shape, dtype=tf.int32)
        multiples = tf.concat([multiples[:parameters_batch_rank], [batch_size], multiples[parameters_batch_rank:]], 0)

        output_shape = tf.concat([shape[:parameters_batch_rank], [batch_size], shape[parameters_batch_rank:]], 0)

        p_tiled = tf.tile(tf.expand_dims(self.p, parameters_batch_rank), multiples)
        
        random_samples = tf.random.uniform(output_shape, minval=0., maxval=1.)
        bernoulli_samples = tf.where(
            random_samples < p_tiled,
            tf.ones(output_shape, dtype=tf.float32),
            tf.zeros(output_shape, dtype=tf.float32)
        )
        
        return bernoulli_samples


class IndependentBernoulliLayer(ParameterizableDistribution):
    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.hidden_layer = tf.keras.layers.Dense(num_dims, activation=tf.nn.tanh)
        self.output_layer = tf.keras.layers.Dense(num_dims, activation=tf.nn.sigmoid)

    def parameterize(self, inputs) -> DiagonalGaussianParameters:
        hidden = self.hidden_layer(inputs)
        bernoulli_parameters = self.output_layer(hidden)
        return IndependentBernoulliParameters(p=bernoulli_parameters)


@dataclass
class DLGMParameters(DistributionParameters):
    layer_parameters: Sequence[DistributionParameters]

    @property
    def parameters_batch_rank(self):
        return self.layer_parameters[0].parameters_batch_rank

    def log_likelihood(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        layer_ll = [x.log_likelihood(y) for x, y in zip(self.layer_parameters, inputs)]
        return tf.reduce_sum(tf.stack(layer_ll, 0), 0)
    
    def sample_batch(self, batch_size: int = 1) -> List[tf.Tensor]:
        return [x.sample_batch(batch_size=batch_size) for x in self.layer_parameters]

    def sample(self) -> List[tf.Tensor]:
        return [x.sample() for x in self.layer_parameters]


class DLGM(ParameterizableDistribution):
    def __init__(self, layers: Sequence[ParameterizableDistribution]):
        super().__init__()
        self.layers = list(layers)

    def parameterize(self, inputs: Sequence[tf.Tensor]):
        return DLGMParameters(layer_parameters=[x.parameterize(y) for x, y in zip(self.layers, inputs)])


@dataclass
class NormalizingFlowParameters(DistributionParameters):
    base_parameters: DistributionParameters
    flow_parameters: Sequence[FlowParameters]

    @property
    def parameters_batch_rank(self):
        return self.base_parameters.parameters_batch_rank

    def log_likelihood(self, inputs: tf.Tensor):
        base_log_likelihood = self.base_parameters.log_likelihood(inputs)
        flow_ll = [x.jacobian_log_determinant(inputs) for x in self.flow_parameters]
        joint_flow_ll = tf.reduce_sum(tf.stack(flow_ll, 0), 0)
        return base_log_likelihood - joint_flow_ll

    def sample_batch(self, batch_size: int):
        batch = self.base_parameters.sample_batch(batch_size=batch_size)
        for fp in self.flow_parameters:
            batch = fp.transform(batch)
        return batch


class NormalizingFlow(ParameterizableDistribution):
    def __init__(self, base_distribution: ParameterizableDistribution, flows: Sequence[Flow]):
        assert len(flows) > 0, 'must use at least 1 flow'
        self.base_distribution = base_distribution
        self.flows = list(flows)

    def parameterize(self, inputs):
        return NormalizingFlowParameters(base_parameters=self.base_distribution.parameterize(inputs), flow_parameters=[
            x.parameterize(inputs) for x in self.flows
        ])
