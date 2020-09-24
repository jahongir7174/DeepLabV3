import os
from os.path import join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from sklearn import metrics
from utils import util, config
import numpy as np
from nets import nn

if __name__ == '__main__':
    D = "Defect"
    N = "Normal"

    true = []
    pred = []

    _, palette = util.get_label_info(os.path.join('..', config.data_dir, "class_dict.csv"))

    model = nn.build_model(classes=len(palette))
    model.load_weights(join('weights', 'model100.h5'))

    print("\n--- Start Testing ---")
    test_paths = []
    root = join('..', config.data_dir)
    for path in [name for name in os.listdir(join(root, 'test_labels')) if name.endswith('.PNG')]:
        test_paths.append(join(root, 'test', path))
    for test_path in test_paths:
        # folder = test_path.split('\\')[-2]
        input_image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        input_image = cv2.resize(input_image, (config.width, config.height))

        input_image = np.expand_dims(input_image, axis=(0, 3))

        output_image = model.predict(input_image / 127.5 - 1.0)
        output_image = np.array(output_image[0, :, :, :])
        output_image = util.reverse_one_hot(output_image)
        output_image = util.colour_code_segmentation(output_image, palette)
        output_image = np.uint8(output_image)

        label_image = cv2.imread(test_path[:-4] + '_L.png', cv2.IMREAD_COLOR)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        label_image = cv2.resize(label_image, (config.width, config.height))

        mask = (label_image == [0, 0, 255]).all(axis=2)

        if True in mask:
            is_defect = True
            true.append(D)
        else:
            is_defect = False
            true.append(N)

        mask = (output_image == [0, 0, 255]).all(axis=2)

        if True in mask:
            is_defect = True
            pred.append(D)
        else:
            is_defect = False
            pred.append(N)
        input_image = np.uint8(np.squeeze(input_image, axis=(0, 3)))
        util.save_images([input_image, label_image, output_image],
                         join(config.result_path, os.path.basename(test_path)[:-4] + '.png'),
                         titles=['INPUT', 'True', 'Pred'])

        print(test_path)

    print(metrics.confusion_matrix(true, pred))
    print(metrics.classification_report(true, pred, digits=3))
