import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Optional

from normalizing_flows.models import NormalizingFlowImageModel
from normalizing_flows.ops import merge_dims

import os
import fire


class NormalizingFlowsTrainer:
    def __init__(self, log_dir: str, dataset_name: str = 'mnist'):
        assert 'mnist' in dataset_name or dataset_name.startswith('cifar')
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.dataset = tfds.load(self.dataset_name,
                                 split='train[:70%]',
                                 shuffle_files=True)
        self.validation_dataset = tfds.load(self.dataset_name,
                                            split='train[70%:]',
                                            shuffle_files=True)
        self.image_shape = self.dataset.element_spec['image'].shape

    def _extract_preprocessed_image(
            self, item, model: NormalizingFlowImageModel) -> tf.Tensor:
        # Get sample.
        images = tf.round(tf.cast(item['image'], dtype=tf.float32) / 255.)
        preprocessed = model.preprocess_images(images)
        return preprocessed

    def _compute_loss(self, preprocessed: tf.Tensor,
                      model: NormalizingFlowImageModel, batch_size: int,
                      sample_size: int, beta: float):
        # Parameterize recognition model.
        recognition_params = model.recognition_model.parameterize(preprocessed)

        # Differentiably sample from posterior.
        recognition_samples = recognition_params.sample_batch(sample_size)

        # Parameterize the generative model.
        generator_params = model.generative_model.parameterize(
            [[batch_size, sample_size], recognition_samples])

        # Compute log-likelihood of approximate posterior.
        approx_posterior_ll = recognition_params.log_likelihood(
            recognition_samples)

        # Compute log-likelihood of generator using latent
        # variables and data samples.
        observation_samples = tf.expand_dims(preprocessed, 1)
        generator_ll = generator_params.log_likelihood(
            [recognition_samples, observation_samples])

        # Formulate the loss.
        loss = -tf.reduce_mean(approx_posterior_ll + beta * generator_ll)

        return loss

    def run(self,
            initial_beta: float = 0.01,
            beta_schedule_duration: int = 10000,
            num_steps: int = 1000,
            batch_size: int = 100,
            summary_decimation: int = 100,
            num_flows: int = 10,
            sample_size: int = 1,
            seed: int = 42,
            shuffle_buffer_size: int = int(1e4),
            learning_rate: float = 1e-5,
            momentum: float = 0.9,
            device_name: Optional[str] = None):
        os.makedirs(self.log_dir, exist_ok=True)
        with tf.device(device_name):
            tf.random.set_seed(seed)
            beta = tf.Variable(initial_beta, dtype=tf.float32, trainable=False)
            beta_schedule_duration = tf.Variable(beta_schedule_duration,
                                                 dtype=tf.int64,
                                                 trainable=False)
            step = tf.Variable(0, dtype=tf.int64, trainable=False)
            num_flows = tf.Variable(num_flows, dtype=tf.int32, trainable=False)

            model = NormalizingFlowImageModel(self.image_shape,
                                              num_flows=int(num_flows))
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate, momentum=momentum)
            ckpt = tf.train.Checkpoint(
                beta=beta,
                beta_schedule_duration=beta_schedule_duration,
                num_flows=num_flows,
                step=step,
                model=model,
                optimizer=optimizer,
            )

            manager = tf.train.CheckpointManager(ckpt, self.log_dir, 10)
            ckpt_path = manager.restore_or_initialize()
            if ckpt_path is not None:
                print('Loaded existing checkpoint', ckpt_path)

            summary_writer = tf.summary.create_file_writer(self.log_dir)
            beta_update = 1. / tf.cast(beta_schedule_duration,
                                       dtype=tf.float32)
            d = self.dataset.repeat().shuffle(shuffle_buffer_size).batch(
                batch_size).take(num_steps)

            val_ds = self.validation_dataset.repeat().shuffle(
                shuffle_buffer_size).batch(batch_size)
            val_ds_iter = iter(val_ds)
            for item in d:
                preprocessed = self._extract_preprocessed_image(item, model)

                with tf.GradientTape() as tape:
                    loss = self._compute_loss(preprocessed, model, batch_size,
                                              sample_size, beta)

                tf.debugging.check_numerics(loss, 'loss has invalid numerics')

                # Perform back propagation.
                trainable_variables = model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                for gix, grad in enumerate(gradients):
                    tf.debugging.check_numerics(
                        grad, f'gradient {gix} has invalid numerics')
                optimizer.apply_gradients(zip(gradients, trainable_variables))

                val_loss = self._compute_loss(
                    self._extract_preprocessed_image(next(val_ds_iter), model),
                    model, batch_size, sample_size, beta)

                step.assign_add(1)
                beta.assign(tf.minimum(1., beta + beta_update))

                # Print info.
                print('Step =', step.numpy(), ', Train Loss =', loss.numpy(),
                      ', Val Loss =', val_loss.numpy())

                # Write summaries.
                if step % summary_decimation == 0:
                    manager.save()
                    with summary_writer.as_default():
                        tf.summary.scalar('loss/train', loss, step=step)
                        tf.summary.scalar('loss/val', val_loss, step=step)

            manager.save()


if __name__ == '__main__':
    fire.Fire(NormalizingFlowsTrainer)
