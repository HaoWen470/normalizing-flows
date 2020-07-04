import tensorflow as tf

from typing import List

from normalizing_flows.distributions import (
    DLGM, NormalizingFlow, DiagonalGaussianLayer,
    StandardGaussianLayer, IndependentBernoulliLayer)

from normalizing_flows.flows import PlanarFlow


class NormalizingFlowImageModel(tf.Module):
    def __init__(self, image_shape: List[int], num_flows: int = 1):
        super().__init__()
        self.image_shape = image_shape
        self.image_size = tf.reduce_prod(self.image_shape)
        self.recognition_model = NormalizingFlow(
            DiagonalGaussianLayer(400),
            [
                PlanarFlow(400) for _ in range(num_flows)
            ])
        
        self.generative_model = DLGM([
            StandardGaussianLayer(400),
            IndependentBernoulliLayer(self.image_size)
        ])
    
    def preprocess_images(self, image):
        image_shape = tf.shape(image)
        output_shape = tf.concat([image_shape[:-3], [self.image_size]], 0)
        return tf.reshape(image, output_shape)

    def postprocess_generative_samples(self, sample):
        sample_shape = tf.shape(sample)
        output_shape = tf.concat([sample_shape[:-1], self.image_shape], 0)
        return tf.reshape(sample, output_shape)
