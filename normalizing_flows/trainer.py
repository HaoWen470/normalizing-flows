import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Optional

from normalizing_flows.models import NormalizingFlowImageModel
from normalizing_flows.ops import merge_dims

import os
import fire


class NormalizingFlowsTrainer:
    def __init__(self, log_dir: str, dataset_name: str):
        assert 'mnist' in dataset_name or dataset_name.startswith('cifar')
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.dataset = tfds.load(self.dataset_name, split='train', shuffle_files=True)
        self.image_shape = self.dataset.element_spec['image'].shape

    def run(self, initial_beta: float = 0.01, beta_schedule_duration: int = 10000,
            num_steps: int = 1000, batch_size: int = 32, summary_decimation: int = 100,
            num_flows: int = 1, sample_size: int = 1, seed: int = 42,
            shuffle_buffer_size: int = int(1e4), 
            device_name: Optional[str] = None):
        os.makedirs(self.log_dir, exist_ok=True)
        with tf.device(device_name):
            tf.random.set_seed(seed)
            beta = tf.Variable(initial_beta, dtype=tf.float32, trainable=False)
            step = tf.Variable(0, dtype=tf.int64, trainable=False)

            model = NormalizingFlowImageModel(self.image_shape, num_flows=num_flows)
            optimizer = tf.keras.optimizers.Adam()
            ckpt = tf.train.Checkpoint(
                beta=beta,
                step=step,
                model=model,
                optimizer=optimizer,
            )

            manager = tf.train.CheckpointManager(ckpt, self.log_dir, 10)
            ckpt_path = manager.restore_or_initialize()
            if ckpt_path is not None:
                print('Loaded existing checkpoint', ckpt_path)
            
            summary_writer = tf.summary.create_file_writer(self.log_dir)
            beta_update = 1. / beta_schedule_duration
            d = self.dataset.shuffle(shuffle_buffer_size).repeat().batch(batch_size).take(num_steps)
            for item in d:
                # Get sample.
                images = tf.round(tf.cast(item['image'], dtype=tf.float32) / 255.)
                preprocessed = model.preprocess_images(images)

                with tf.GradientTape() as tape:
                    # Parameterize recognition model.
                    recognition_params = model.recognition_model.parameterize(preprocessed)

                    # Differentiably sample from posterior.
                    recognition_samples = recognition_params.sample_batch(sample_size)

                    # Parameterize the generative model.
                    generator_params = model.generative_model.parameterize([
                        [batch_size, sample_size],
                        recognition_samples])

                    # Compute log-likelihood of approximate posterior.
                    approx_posterior_ll = recognition_params.log_likelihood(recognition_samples)

                    # Compute log-likelihood of generator using latent variables and data samples.
                    observation_samples = tf.expand_dims(preprocessed, 1)
                    generator_ll = generator_params.log_likelihood([recognition_samples, observation_samples])

                    # Formulate the loss.
                    loss = -tf.reduce_mean(approx_posterior_ll + beta * generator_ll / tf.cast(sample_size, tf.float32))

                # Perform back propagation.
                trainable_variables = model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))

                # Write summaries.
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)

                # Print info.
                print('Step =', step.numpy(), ', Loss =', loss.numpy())

                step.assign_add(1)
                beta.assign(tf.minimum(1., beta + beta_update))
                if step % summary_decimation == 0:
                    manager.save()

            manager.save()


if __name__ == '__main__':
    fire.Fire(NormalizingFlowsTrainer)