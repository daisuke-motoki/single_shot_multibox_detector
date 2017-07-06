import os
import pickle
import numpy as np
from ssd.ssd import SingleShotMultiBoxDetector
from generator import Generator


if __name__ == "__main__":
    # settings
    labels = ["aeroplane",
              "bus", "bicycle", "bird", "boat", "bottle",
              "car", "cat", "chair", "cow",
              "diningtable", "dog",
              "horse",
              "motorbike",
              "pottedplant", "person",
              "sheep", "sofa",
              "train", "tvmonitor"]
    n_classes = len(labels)
    input_shape = (300, 300, 3)

    # ssd keras conditions
    aspect_ratios = [[2., 1/2.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.]]
    scales = [(30.,),
              (60., 114.),
              (114., 168.),
              (168., 222.),
              (222., 276.),
              (276., 330.)]
    variances = [0.1, 0.1, 0.2, 0.2]

    # create network
    ssd = SingleShotMultiBoxDetector(model_type="ssd300",
                                     base_net="vgg16",
                                     n_classes=n_classes,
                                     input_shape=input_shape,
                                     aspect_ratios=aspect_ratios,
                                     scales=scales,
                                     variances=variances,
                                     overlap_threshold=0.5,
                                     nms_threshold=0.45,
                                     max_output_size=400)
    # ssd.build(init_weight="./checkpoints/weights.29-1.16.hdf5")
    ssd.build()

    # make generator for training images
    batch_size = 8
    GROUND_TRUTH = "voc2007_annotations.pkl"
    INPUT_DATA = "../../../share/image-net/VOCdevkit/VOC2007/JPEGImages"
    gt = pickle.load(open(GROUND_TRUTH, 'rb'))
    keys = sorted(gt.keys())
    indexes = np.arange(len(keys))
    np.random.seed(0)
    np.random.shuffle(indexes)
    num_train = int(round(0.8 * len(keys)))

    train_keys = np.array(keys)[indexes[:num_train]].tolist()
    val_keys = np.array(keys)[indexes[num_train:]].tolist()
    num_val = len(val_keys)
    gen = Generator(
        gt, ssd.bboxes, batch_size, INPUT_DATA,
        train_keys, val_keys,
        (input_shape[0], input_shape[1]),
        saturation_var=0.5, brightness_var=0.5,
        contrast_var=0.5, lighting_std=0.5,
        hflip_prob=0.5, vflip_prob=0.5,
        do_crop=False
    )

    # training
    path_to_checkpoints = os.sep.join((
        "./checkpoints",
        "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    ))
    freeze = ["input_1",
              "block1_conv1", "block1_conv2", "block1_pool",
              "block2_conv1", "block2_conv2", "block2_pool",
              # "block3_conv1", "block3_conv2", "block3_conv3", "block3_pool",
              ]
    ssd.train_by_generator(gen,
                           epoch=30,
                           learning_rate=3e-4,
                           neg_pos_ratio=2.0,
                           freeze=freeze,
                           checkpoints=path_to_checkpoints)
    ssd.save_parameters("ssd300_voc2007_params.pkl")