import numpy as np
import tensorflow as tf

from utils import config, util


def input_fn(data_paths, palette):

    def generator_fn():
        counter = 0
        while 1:
            if counter >= len(data_paths[0]):
                counter = 0
            image_path, label_path = data_paths[0][counter], data_paths[1][counter]
            image = util.load_image(image_path)
            label = util.load_image(label_path)
            image, label = util.preprocess_data(image, label, image_path)
            one_hot = util.one_hot(label, palette)
            counter += 1
            yield np.float32(image) / 255.0, np.float32(one_hot)

    dataset = tf.data.Dataset.from_generator(generator_fn,
                                             (tf.float32, tf.float32),
                                             (tf.TensorShape([config.height, config.width, 3]),
                                              tf.TensorShape([config.height, config.width, len(palette)])))

    dataset = dataset.batch(config.batch_size)
    dataset = dataset.repeat(config.epochs)
    dataset = dataset.prefetch(40 * config.batch_size)
    return dataset
