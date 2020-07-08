import tensorflow as tf
import tensorflow_datasets as tfds

from normalizing_flows.models import NormalizingFlowImageModel
from normalizing_flows.ops import merge_dims

from typing import Optional

import imageio
import fire


def main(*,
         log_dir: str,
         dataset_name: str,
         num_flows: int,
         output_path: str,
         max_frames: int = 100,
         fps: float = 10.,
         batch_size: int = 32,
         device_name: Optional[str] = None):
    assert output_path.lower().endswith(
        '.gif'), 'must specify a GIF output path'
    with tf.device(device_name):
        dataset = tfds.load(dataset_name, split='test', shuffle_files=True)
        image_shape = dataset.element_spec['image'].shape
        model = NormalizingFlowImageModel(image_shape, num_flows=num_flows)
        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, log_dir, None)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

        d = dataset.take(max_frames).batch(32)
        samples = []
        for item in d:
            images = tf.round(tf.cast(item['image'], dtype=tf.float32) / 255.)
            preprocessed = model.preprocess_images(images)

            # Parameterize the approximate posterior (recognition network).
            recognition_params = model.recognition_model.parameterize(
                preprocessed)

            # Sample from q(z | x).
            recognition_sample = recognition_params.sample()

            # Parameterize p(x | z).
            generative_params = model.generative_model.parameterize(
                [[recognition_sample.shape[0]], recognition_sample])

            # Sample from p(x | z).
            reconstruction_params = generative_params.layer_parameters[-1]
            reconstructions = reconstruction_params.sample()
            reconstruction_images = model.postprocess_generative_samples(
                reconstructions)

            # Create demo image (left is original, right is reconstruction).
            demo_images = tf.concat([images, reconstruction_images], axis=2)

            # Append samples.
            samples.append(demo_images)

        if samples:
            video = tf.concat(samples, 0)
            imageio.mimwrite(output_path, video, loop=0, fps=fps)
            print(f'Saved samples to {output_path}')
        else:
            print('No samples generated')


if __name__ == '__main__':
    fire.Fire(main)
