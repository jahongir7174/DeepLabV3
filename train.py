import os
import sys
from os.path import exists
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from nets import nn
from utils import config, util, data_loader

np.random.seed(config.seed)
tf.random.set_seed(config.seed)

_, palette = util.get_label_info(join(config.data_dir, 'class_dict.csv'))

names = [name for name in os.listdir(join(config.data_dir, config.image_dir))]
image_paths = []
label_paths = []
for name in names:
    image_path = join(config.data_dir, config.image_dir, name)
    label_path = join(config.data_dir, config.label_dir, name[:-4]) + '_L' + name[-4:]
    if exists(image_path) and exists(label_path):
        image_paths.append(image_path)
        label_paths.append(label_path)

strategy = tf.distribute.MirroredStrategy()
nb_gpu = strategy.num_replicas_in_sync

dataset = data_loader.input_fn((image_paths, label_paths), palette)
dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = nn.build_model(classes=len(palette))
    model_name = 'efficientnet-b7'
    file_name = f'{model_name}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    file_hash = nn.WEIGHTS_HASHES[model_name[-2:]][1]
    weights_path = get_file(file_name,
                            nn.BASE_WEIGHTS_PATH + file_name,
                            cache_subdir='models',
                            file_hash=file_hash)
    model.load_weights(weights_path, by_name=True)

print('[INFO] {} train data'.format(len(image_paths)))

with strategy.scope():
    loss_object = nn.segmentation_loss


    def compute_loss(y_true, y_pred):
        per_example_loss = loss_object(y_true, y_pred)
        scale_loss = tf.reduce_sum(per_example_loss) * 1. / nb_gpu

        return scale_loss

with strategy.scope():
    def train_step(image, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(image)
            loss = compute_loss(y_true, y_pred)
        train_variable = model.trainable_variables
        gradient = tape.gradient(loss, train_variable)
        optimizer.apply_gradients(zip(gradient, train_variable))

        return loss

with strategy.scope():
    @tf.function
    def distribute_train_step(image, y_true):
        loss = strategy.experimental_run_v2(train_step, args=(image, y_true))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)


def main():
    nb_steps = len(image_paths) // (nb_gpu * config.batch_size)
    print("--- Training with {} Steps ---".format(nb_steps))
    if not exists('weights'):
        os.makedirs('weights')
    for step, inputs in enumerate(dataset):
        step += 1
        image, y_true = inputs
        loss = distribute_train_step(image, y_true)
        print(f'{step} - {loss.numpy():.6f}')
        if step % nb_steps == 0:
            model.save_weights(join("weights", "model{}.h5".format(step // nb_steps)))
        if step // nb_steps == config.epochs:
            sys.exit("--- Stop Training ---")


if __name__ == '__main__':
    main()
