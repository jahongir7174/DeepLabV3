from __future__ import print_function, division

import csv
import multiprocessing
import os
from os.path import exists
from os.path import join
import random
import cv2
import numpy as np

from utils import config


def load_image(path):
    return cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)


def write_image(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def get_label_info(csv_path):
    class_names = []
    label_values = []
    with open(csv_path, 'r') as reader:
        file_reader = csv.reader(reader, delimiter=',')
        next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    label_values = np.array(label_values)
    x = label_values[image.astype(int)]
    return x


def save_images(images, path, cols=1, titles=None):
    from matplotlib import pyplot
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    pyplot.axis('off')
    fig = pyplot.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            pyplot.gray()
        pyplot.imshow(image)
        pyplot.axis("off")
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    pyplot.savefig(path)
    pyplot.close(fig)


def calculate_class_weight(label_file):
    _, palette = get_label_info(join(config.data_dir, 'class_dict.csv'))
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0

    image = load_image(label_file)
    for i, label in enumerate(palette):
        class_mask = (image == label)
        class_mask = np.all(class_mask, axis=2)
        class_mask = class_mask.astype(np.float32)
        class_frequency = np.sum(class_mask)
        label_to_frequency[i] += class_frequency
    return label_to_frequency


def get_class_weights():
    paths = [name for name in os.listdir(join(config.data_dir, config.image_dir))]
    np.random.shuffle(paths)

    train_labels = []
    for path in paths:
        image_path = join(config.data_dir, config.image_dir, path)
        label_path = join(config.data_dir, config.label_dir, path[:-4] + '_L' + path[-4:])
        if exists(image_path) and exists(label_path):
            train_labels.append(label_path)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        frequencies = pool.map(calculate_class_weight, train_labels)
    pool.close()
    _, palette = get_label_info(join(config.data_dir, 'class_dict.csv'))
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0

    for frequency in frequencies:
        label_to_frequency[0] += frequency[0]
        label_to_frequency[1] += frequency[1]
        label_to_frequency[2] += frequency[2]

    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)
    class_weights = np.array(class_weights, np.float32)
    return class_weights


def random_crop(image, label, crop_height, crop_width, path):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        print(path, '---', image.shape, '---', label.shape)
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
            crop_height, crop_width, image.shape[0], image.shape[1]))


def one_hot(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def preprocess_data(input_image, output_image, path):
    return random_crop(input_image, output_image, config.height, config.width, path)
