from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models

BASE_WEIGHTS_PATH = 'https://github.com/Callidior/keras-applications/releases/download/efficientnet/'
WEIGHTS_HASHES = {'b0': ('e9e877068bd0af75e0a36691e03c072c', '345255ed8048c2f22c793070a9c1a130'),
                  'b1': ('8f83b9aecab222a9a2480219843049a1', 'b20160ab7b79b7a92897fcb33d52cc61'),
                  'b2': ('b6185fdcd190285d516936c09dceeaa4', 'c6e46333e8cddfa702f4d8b8b6340d70'),
                  'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb', 'e0cf8654fad9d3625190e30d70d0c17d'),
                  'b4': ('ab314d28135fe552e2f9312b31da6926', 'b46702e4754d2022d62897e0618edc7b'),
                  'b5': ('8d60b903aff50b09c6acf8eaba098e09', '0a839ac36e46552a881f2975aaab442f'),
                  'b6': ('a967457886eac4f5ab44139bdd827920', '375a35c17ef70d46f9c664b03b4437f2'),
                  'b7': ('e964fd6e26e9a4c144bcb811f2a10f20', 'd55674cc46b805f4382d18bc08ed43c1')}

DEFAULT_BLOCKS_ARGS = [{'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
                        'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
                       {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
                        'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
                       {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
                        'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
                       {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
                        'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
                       {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
                        'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
                       {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
                        'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
                       {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
                        'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}]

CONV_KERNEL_INITIALIZER = {'class_name': 'VarianceScaling',
                           'config': {'scale': 2.0, 'mode': 'fan_out', 'distribution': 'normal'}}

DENSE_KERNEL_INITIALIZER = {'class_name': 'VarianceScaling',
                            'config': {'scale': 1. / 3., 'mode': 'fan_out', 'distribution': 'uniform'}}


def correct_pad(inputs, kernel_size):
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1])


def round_filters(filters, depth_divisor, width_coefficient):
    filters *= width_coefficient
    new_filters = max(depth_divisor, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))


def activation_fn(x):
    return tf.nn.swish(x)


def block(inputs, drop_rate=0., name='', filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    bn_axis = 3

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depth-wise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(correct_pad(x, (kernel_size, kernel_size)), name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)

        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv2D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip is True and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, (None, 1, 1, 1), name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')

    return x


def efficient_net(width_coefficient, depth_coefficient, input_tensor=None):
    depth_divisor = 8
    drop_connect_rate = 0.2
    blocks_args = DEFAULT_BLOCKS_ARGS

    features = []
    img_input = input_tensor

    bn_axis = 3

    # Build stem
    x = img_input
    x = layers.ZeroPadding2D(padding=correct_pad(x, (3, 3)), name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32, depth_divisor, width_coefficient), 3, 2, 'valid',
                      use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation_fn, name='stem_activation')(x)

    # Build blocks
    from copy import deepcopy
    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'], depth_divisor, width_coefficient)
        args['filters_out'] = round_filters(args['filters_out'], depth_divisor, width_coefficient)

        for j in range(round_repeats(args.pop('repeats'), depth_coefficient)):
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, drop_connect_rate * b / blocks, name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1
        if i < len(blocks_args) - 1 and blocks_args[i + 1]['strides'] == 2:
            features.append(x)
        elif i == len(blocks_args) - 1:
            features.append(x)

    return features


def efficient_net_b0(input_tensor=None):
    return efficient_net(1.0, 1.0, input_tensor=input_tensor)


def efficient_net_b1(input_tensor=None):
    return efficient_net(1.0, 1.1, input_tensor=input_tensor)


def efficient_net_b2(input_tensor=None):
    return efficient_net(1.1, 1.2, input_tensor=input_tensor)


def efficient_net_b3(input_tensor=None):
    return efficient_net(1.2, 1.4, input_tensor=input_tensor)


def efficient_net_b4(input_tensor=None):
    return efficient_net(1.4, 1.8, input_tensor=input_tensor)


def efficient_net_b5(input_tensor=None):
    return efficient_net(1.6, 2.2, input_tensor=input_tensor)


def efficient_net_b6(input_tensor=None):
    return efficient_net(1.8, 2.6, input_tensor=input_tensor)


def efficient_net_b7(input_tensor=None):
    return efficient_net(2.0, 3.1, input_tensor=input_tensor)


def segmentation_loss(y_true, y_pred):
    return tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred)


def separable_bn(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.Activation(tf.nn.relu)(x)
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), (stride, stride), dilation_rate=(rate, rate),
                               padding=depth_padding, use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation(tf.nn.relu)(x)
    x = layers.Conv2D(filters, (1, 1), 1, 'same', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation(tf.nn.relu)(x)

    return x


def build_model(input_shape=(512, 512, 3), classes=3):
    inputs = layers.Input(shape=input_shape)

    rates = (6, 12, 18)

    features = efficient_net_b7(inputs)
    x, skip1 = features[-2], features[1]

    b0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization(epsilon=1e-5)(b0)
    b0 = layers.Activation(tf.nn.relu)(b0)

    b1 = separable_bn(x, 256, rate=rates[0], depth_activation=True, epsilon=1e-5)
    b2 = separable_bn(x, 256, rate=rates[1], depth_activation=True, epsilon=1e-5)
    b3 = separable_bn(x, 256, rate=rates[2], depth_activation=True, epsilon=1e-5)

    size_before = tf.keras.backend.int_shape(x)
    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Lambda(lambda _x: backend.expand_dims(_x, 1))(b4)
    b4 = layers.Lambda(lambda _x: backend.expand_dims(_x, 1))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(b4)
    b4 = layers.BatchNormalization(epsilon=1e-5)(b4)
    b4 = layers.Activation(tf.nn.relu)(b4)
    b4 = layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                            size_before[1:3],
                                                            method='bilinear',
                                                            align_corners=True))(b4)

    x = layers.concatenate([b4, b0, b1, b2, b3])

    x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(tf.nn.relu)(x)
    x = layers.Dropout(0.1)(x)

    skip_size = backend.int_shape(skip1)
    x = layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                           skip_size[1:3],
                                                           method='bilinear',
                                                           align_corners=True))(x)

    dec_skip1 = layers.Conv2D(48, (1, 1), padding='same', use_bias=False)(skip1)
    dec_skip1 = layers.BatchNormalization(epsilon=1e-5)(dec_skip1)
    dec_skip1 = layers.Activation(tf.nn.relu)(dec_skip1)

    x = layers.concatenate([x, dec_skip1])

    x = separable_bn(x, 256, depth_activation=True, epsilon=1e-5)
    x = separable_bn(x, 256, depth_activation=True, epsilon=1e-5)

    x = layers.Conv2D(classes, (1, 1), padding='same')(x)

    size_before3 = backend.int_shape(inputs)
    x = layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                           size_before3[1:3],
                                                           method='bilinear',
                                                           align_corners=True))(x)

    return models.Model(inputs, x)
