import os
from ssd.ssd import SingleShotMultiBoxDetector
from generators import RandomMahJangGenerator


if __name__ == "__main__":
    # settings
    pi_names = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
        "c", "e", "f", "h", "n", "s", "w",
    ]
    n_classes = len(pi_names)
    input_shape = (512, 512, 3)
    aspect_ratios = [[2., 1/2.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2., 3., 1/3.],
                     [2., 1/2.],
                     [2., 1/2.]]
    scales = [(30., 60.),
              (60., 111.),
              (111., 162.),
              (162., 213.),
              (213., 264.),
              (264., 315.)]
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
    ssd.build()

    # make generator for training images
    batch_size = 8
    n_sample = batch_size*100
    val_sample = int(n_sample*0.2)
    PI_IMAGES = os.sep.join((
        "/home/daisuke/work/mah_jang/data",
        "images",
        "pi/"
    ))
    gen = RandomMahJangGenerator(
        ssd.bboxes, n_sample, val_sample, batch_size, PI_IMAGES,
        (input_shape[0], input_shape[1]), pi_names,
        saturation_var=0.5, brightness_var=0.5,
        contrast_var=0.5, lighting_std=0.5,
        hflip_prob=0.5, vflip_prob=0.5,
        do_crop=False
    )

    # training
    path_to_checkpoints = os.sep.join((
        "/home/daisuke/work/single_shot_multibox_detector/data/models/checkpoints",
        "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    ))
    freeze = ["input_1",
              "block1_conv1", "block1_conv2", "block1_pool",
              "block2_conv1", "block2_conv2", "block2_pool",
              "block3_conv1", "block3_conv2", "block3_conv3", "block3_pool",
              ]
    ssd.train_by_generator(gen,
                           epoch=30,
                           learning_rate=1e-4,
                           neg_pos_ratio=3.0,
                           freeze=freeze,
                           checkpoints=path_to_checkpoints)
    ssd.save_parameters("ssd300_params.pkl")
