import numpy as np
from collections import OrderedDict
from keras.layers import Input, Conv2D, MaxPool2D
from keras.layers import Flatten, Reshape, Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from ssd.layers import L2Normalization
from ssd.utils import make_bboxes


AVAILABLE_TYPE = ["300"]
AVAILABLE_BASENET = ["vgg16"]


def _build_resnet50_basenet(network):
    """
    """
    from keras.layers import Activation
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import ZeroPadding2D
    from keras.layers import BatchNormalization
    import keras.backend as K
    from keras.applications.resnet50 import conv_block, identity_block
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = ZeroPadding2D((3, 3))(network["input"])
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    network["conv_block2"] = conv_block(x, 3, [64, 64, 256],
                                        stage=2, block='a', strides=(1, 1))
    network["identity_block2_1"] = identity_block(network["conv_block2"],
                                                  3, [64, 64, 256],
                                                  stage=2, block='b')
    network["identity_block2_2"] = identity_block(network["identity_block2_1"],
                                                  3, [64, 64, 256],
                                                  stage=2, block='c')

    network["conv_block3"] = conv_block(network["identity_block2_2"],
                                        3, [128, 128, 512],
                                        stage=3, block='a')
    network["identity_block3_1"] = identity_block(network["conv_block3"],
                                                  3, [128, 128, 512],
                                                  stage=3, block='b')
    network["identity_block3_2"] = identity_block(network["identity_block3_1"],
                                                  3, [128, 128, 512],
                                                  stage=3, block='c')
    network["identity_block3_3"] = identity_block(network["identity_block3_2"],
                                                  3, [128, 128, 512],
                                                  stage=3, block='d')

    network["conv_block4"] = conv_block(network["identity_block3_3"],
                                        3, [256, 256, 1024],
                                        stage=4, block='a')
    network["identity_block4_1"] = identity_block(network["conv_block4"],
                                                  3, [256, 256, 1024],
                                                  stage=4, block='b')
    network["identity_block4_2"] = identity_block(network["identity_block4_1"],
                                                  3, [256, 256, 1024],
                                                  stage=4, block='c')
    network["identity_block4_3"] = identity_block(network["identity_block4_2"],
                                                  3, [256, 256, 1024],
                                                  stage=4, block='d')
    network["identity_block4_4"] = identity_block(network["identity_block4_3"],
                                                  3, [256, 256, 1024],
                                                  stage=4, block='e')
    network["identity_block4_5"] = identity_block(network["identity_block4_4"],
                                                  3, [256, 256, 1024],
                                                  stage=4, block='f')

    network["conv_block5"] = conv_block(network["identity_block4_5"],
                                        3, [512, 512, 2048],
                                        stage=5, block='a')
    network["identity_block5_1"] = identity_block(network["conv_block5"],
                                                  3, [512, 512, 2048],
                                                  stage=5, block='b')
    network["identity_block5_2"] = identity_block(network["identity_block5_1"],
                                                  3, [512, 512, 2048],
                                                  stage=5, block='c')
    return network


def _build_vgg16_basenet(network):
    """
    """
    # convolution layer 1
    network["block1_conv1"] = Conv2D(64, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block1_conv1"
                                     )(network["input"])
    network["block1_conv2"] = Conv2D(64, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block1_conv2"
                                     )(network["block1_conv1"])
    network["block1_pool"] = MaxPool2D((2, 2),
                                       strides=(2, 2),
                                       padding="same",
                                       name="block1_pool"
                                       )(network["block1_conv2"])
    # convlution layer 2
    network["block2_conv1"] = Conv2D(128, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block2_conv1"
                                     )(network["block1_pool"])
    network["block2_conv2"] = Conv2D(128, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block2_conv2"
                                     )(network["block2_conv1"])
    network["block2_pool"] = MaxPool2D((2, 2),
                                       strides=(2, 2),
                                       padding="same",
                                       name="block2_pool"
                                       )(network["block2_conv2"])
    # convlution layer 3
    network["block3_conv1"] = Conv2D(256, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block3_conv1"
                                     )(network["block2_pool"])
    network["block3_conv2"] = Conv2D(256, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block3_conv2"
                                     )(network["block3_conv1"])
    network["block3_conv3"] = Conv2D(256, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block3_conv3"
                                     )(network["block3_conv2"])
    network["block3_pool"] = MaxPool2D((2, 2),
                                       strides=(2, 2),
                                       padding="same",
                                       name="block3_pool"
                                       )(network["block3_conv3"])
    # convlution layer 4
    network["block4_conv1"] = Conv2D(512, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block4_conv1"
                                     )(network["block3_pool"])
    network["block4_conv2"] = Conv2D(512, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block4_conv2"
                                     )(network["block4_conv1"])
    network["block4_conv3"] = Conv2D(512, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block4_conv3"
                                     )(network["block4_conv2"])
    network["block4_pool"] = MaxPool2D((2, 2),
                                       strides=(2, 2),
                                       padding="same",
                                       name="block4_pool"
                                       )(network["block4_conv3"])
    # convlution layer 5
    network["block5_conv1"] = Conv2D(512, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block5_conv1"
                                     )(network["block4_pool"])
    network["block5_conv2"] = Conv2D(512, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block5_conv2"
                                     )(network["block5_conv1"])
    network["block5_conv3"] = Conv2D(512, (3, 3),
                                     activation="relu",
                                     padding="same",
                                     name="block5_conv3"
                                     )(network["block5_conv2"])
    network["block5_pool"] = MaxPool2D((3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       name="block5_pool"
                                       )(network["block5_conv3"])


def SSD300(input_shape, n_classes, basenet_name,
           aspect_ratios, scales):
    """
    """
    network = OrderedDict()
    network["input"] = Input(shape=input_shape)

    # base network -------------------------------------------------------
    if basenet_name == "vgg16":
        _build_vgg16_basenet(network)
    else:
        raise NameError(
            "{} is not defined. Please select from {}".format(
                basenet_name, AVAILABLE_BASENET
            )
        )

    # convolution layer 6 (fc6)
    network["block6_conv"] = Conv2D(1024, (3, 3),
                                    dilation_rate=(6, 6),
                                    activation="relu",
                                    padding="same",
                                    name="block6_conv"
                                    )(network["block5_pool"])
    # convolution layer 7 (fc7)
    network["block7_conv"] = Conv2D(1024, (1, 1),
                                    activation="relu",
                                    padding="same",
                                    name="block7_conv"
                                    )(network["block6_conv"])

    # extra feature layer ---------------------------------------------
    # convolution layer 8
    network["block8_conv1"] = Conv2D(256, (1, 1),
                                     activation="relu",
                                     padding="same",
                                     name="block8_conv1"
                                     )(network["block7_conv"])
    network["block8_conv2"] = Conv2D(512, (3, 3),
                                     strides=(2, 2),
                                     activation="relu",
                                     padding="same",
                                     name="block8_conv2"
                                     )(network["block8_conv1"])
    # convolution layer 9
    network["block9_conv1"] = Conv2D(128, (1, 1),
                                     activation="relu",
                                     padding="same",
                                     name="block9_conv1"
                                     )(network["block8_conv2"])
    network["block9_conv2"] = Conv2D(256, (3, 3),
                                     strides=(2, 2),
                                     activation="relu",
                                     padding="same",
                                     name="block9_conv2"
                                     )(network["block9_conv1"])
    # convolution layer 10
    network["block10_conv1"] = Conv2D(128, (1, 1),
                                      activation="relu",
                                      padding="same",
                                      name="block10_conv1"
                                      )(network["block9_conv2"])
    network["block10_conv2"] = Conv2D(256, (3, 3),
                                      strides=(1, 1),
                                      activation="relu",
                                      padding="valid",
                                      name="block10_conv2"
                                      )(network["block10_conv1"])
    # convolution layer 11
    network["block11_conv1"] = Conv2D(128, (1, 1),
                                      activation="relu",
                                      padding="same",
                                      name="block11_conv1"
                                      )(network["block10_conv2"])
    network["block11_conv2"] = Conv2D(256, (3, 3),
                                      strides=(1, 1),
                                      activation="relu",
                                      padding="valid",
                                      name="block11_conv2"
                                      )(network["block11_conv1"])
    # network["block11_conv2"] = GlobalAveragePooling2D(
    #     name="block11_conv2")(network["block11_conv1"])
    # extra feature layer --------------------------------------------

    # classifier -----------------------------------------------------
    # block4 l2 normalization
    network["block4_norm"] = L2Normalization(20,
                                             name="block4_norm"
                                             )(network["block4_conv3"])

    feature_layers = ["block4_norm", "block7_conv", "block8_conv2",
                      "block9_conv2", "block10_conv2", "block11_conv2"]

    bboxes = list()
    list_loc_layers = list()
    list_conf_layers = list()
    for i in range(len(feature_layers)):
        # make boundary boxes
        layer_name = feature_layers[i]
        aspect_ratio = aspect_ratios[i]
        scale = scales[i]
        feature_map_shape = network[layer_name]._keras_shape
        bbox = make_bboxes(input_shape,
                           feature_map_shape,
                           aspect_ratio,
                           scale)
        bboxes.append(bbox)
        n_boxes = int(len(bbox)/feature_map_shape[1]/feature_map_shape[2])

        # make classifier layers
        layer_name_n = layer_name + "_{}".format(n_classes)
        network[layer_name_n+"_loc"] = Conv2D(n_boxes * 4, (3, 3),
                                              padding="same",
                                              name=layer_name_n+"_loc"
                                              )(network[layer_name])
        network[layer_name_n+"_loc_flat"] = Flatten(
            name=layer_name_n+"_loc_flat")(network[layer_name_n+"_loc"])
        network[layer_name_n+"_conf"] = Conv2D(n_boxes * n_classes, (3, 3),
                                               padding="same",
                                               name=layer_name_n+"_conf"
                                               )(network[layer_name])
        network[layer_name_n+"_conf_flat"] = Flatten(
            name=layer_name_n+"_conf_flat")(network[layer_name_n+"_conf"])

        list_loc_layers.append(network[layer_name_n+"_loc_flat"])
        list_conf_layers.append(network[layer_name_n+"_conf_flat"])
    # classifier -----------------------------------------------------

    # collect predictions
    network["loc"] = concatenate(list_loc_layers,
                                 axis=1,
                                 name="loc")
    network["conf"] = concatenate(list_conf_layers,
                                  axis=1,
                                  name="conf")
    n_all_boxes = network["loc"]._keras_shape[-1] // 4
    network["predictions_loc"] = Reshape((n_all_boxes, 4),
                                         name="predictions_loc"
                                         )(network["loc"])
    reshaped_conf = Reshape((n_all_boxes, n_classes),
                            name="reshaped_conf"
                            )(network["conf"])
    network["predictions_conf"] = Activation("softmax",
                                             name="predictions_conf"
                                             )(reshaped_conf)
    network["predictions"] = concatenate([network["predictions_loc"],
                                          network["predictions_conf"]],
                                         axis=2,
                                         name="predictions")
    # model
    model = Model(network["input"], network["predictions"])
    # bbox
    all_bboxes = np.concatenate(bboxes, axis=0).astype("float32")
    return model, all_bboxes


def SSD512(input_shape, n_classes, basenet_name,
           aspect_ratios, scales):
    """
    """
    network = OrderedDict()
    network["input"] = Input(shape=input_shape)

    # base network -------------------------------------------------------
    if basenet_name == "vgg16":
        _build_vgg16_basenet(network)
    else:
        raise NameError(
            "{} is not defined. Please select from {}".format(
                basenet_name, AVAILABLE_BASENET
            )
        )

    # convolution layer 6 (fc6)
    network["block6_conv"] = Conv2D(1024, (3, 3),
                                    dilation_rate=(6, 6),
                                    activation="relu",
                                    padding="same",
                                    name="block6_conv"
                                    )(network["block5_pool"])
    # convolution layer 7 (fc7)
    network["block7_conv"] = Conv2D(1024, (1, 1),
                                    activation="relu",
                                    padding="same",
                                    name="block7_conv"
                                    )(network["block6_conv"])

    # extra feature layer ---------------------------------------------
    # convolution layer 8
    network["block8_conv1"] = Conv2D(256, (1, 1),
                                     activation="relu",
                                     padding="same",
                                     name="block8_conv1"
                                     )(network["block7_conv"])
    network["block8_conv2"] = Conv2D(512, (3, 3),
                                     strides=(2, 2),
                                     activation="relu",
                                     padding="same",
                                     name="block8_conv2"
                                     )(network["block8_conv1"])
    # convolution layer 9
    network["block9_conv1"] = Conv2D(128, (1, 1),
                                     activation="relu",
                                     padding="same",
                                     name="block9_conv1"
                                     )(network["block8_conv2"])
    network["block9_conv2"] = Conv2D(256, (3, 3),
                                     strides=(2, 2),
                                     activation="relu",
                                     padding="same",
                                     name="block9_conv2"
                                     )(network["block9_conv1"])
    # convolution layer 10
    network["block10_conv1"] = Conv2D(128, (1, 1),
                                      activation="relu",
                                      padding="same",
                                      name="block10_conv1"
                                      )(network["block9_conv2"])
    network["block10_conv2"] = Conv2D(256, (3, 3),
                                      strides=(2, 2),
                                      activation="relu",
                                      padding="same",
                                      name="block10_conv2"
                                      )(network["block10_conv1"])
    # convolution layer 11
    network["block11_conv1"] = Conv2D(128, (1, 1),
                                      activation="relu",
                                      padding="same",
                                      name="block11_conv1"
                                      )(network["block10_conv2"])
    network["block11_conv2"] = Conv2D(256, (3, 3),
                                      strides=(1, 1),
                                      activation="relu",
                                      padding="valid",
                                      name="block11_conv2"
                                      )(network["block11_conv1"])
    # convolution layer 12
    network["block12_conv1"] = Conv2D(128, (1, 1),
                                      activation="relu",
                                      padding="same",
                                      name="block12_conv1"
                                      )(network["block11_conv2"])
    network["block12_conv2"] = Conv2D(256, (2, 2),
                                      strides=(1, 1),
                                      activation="relu",
                                      padding="valid",
                                      name="block12_conv2"
                                      )(network["block12_conv1"])
    # extra feature layer --------------------------------------------

    # classifier -----------------------------------------------------
    # block4 l2 normalization
    network["block4_norm"] = L2Normalization(20,
                                             name="block4_norm"
                                             )(network["block4_conv3"])

    feature_layers = ["block4_norm", "block7_conv", "block8_conv2",
                      "block9_conv2", "block10_conv2", "block11_conv2",
                      "block12_conv2"]

    bboxes = list()
    list_loc_layers = list()
    list_conf_layers = list()
    for i in range(len(feature_layers)):
        # make boundary boxes
        layer_name = feature_layers[i]
        aspect_ratio = aspect_ratios[i]
        scale = scales[i]
        feature_map_shape = network[layer_name]._keras_shape
        bbox = make_bboxes(input_shape,
                           feature_map_shape,
                           aspect_ratio,
                           scale)
        bboxes.append(bbox)
        n_boxes = int(len(bbox)/feature_map_shape[1]/feature_map_shape[2])

        # make classifier layers
        layer_name_n = layer_name + "_{}".format(n_classes)
        network[layer_name_n+"_loc"] = Conv2D(n_boxes * 4, (3, 3),
                                              padding="same",
                                              name=layer_name_n+"_loc"
                                              )(network[layer_name])
        network[layer_name_n+"_loc_flat"] = Flatten(
            name=layer_name_n+"_loc_flat")(network[layer_name_n+"_loc"])
        network[layer_name_n+"_conf"] = Conv2D(n_boxes * n_classes, (3, 3),
                                               padding="same",
                                               name=layer_name_n+"_conf"
                                               )(network[layer_name])
        network[layer_name_n+"_conf_flat"] = Flatten(
            name=layer_name_n+"_conf_flat")(network[layer_name_n+"_conf"])

        list_loc_layers.append(network[layer_name_n+"_loc_flat"])
        list_conf_layers.append(network[layer_name_n+"_conf_flat"])
    # classifier -----------------------------------------------------

    # collect predictions
    network["loc"] = concatenate(list_loc_layers,
                                 axis=1,
                                 name="loc")
    network["conf"] = concatenate(list_conf_layers,
                                  axis=1,
                                  name="conf")
    n_all_boxes = network["loc"]._keras_shape[-1] // 4
    network["predictions_loc"] = Reshape((n_all_boxes, 4),
                                         name="predictions_loc"
                                         )(network["loc"])
    reshaped_conf = Reshape((n_all_boxes, n_classes),
                            name="reshaped_conf"
                            )(network["conf"])
    network["predictions_conf"] = Activation("softmax",
                                             name="predictions_conf"
                                             )(reshaped_conf)
    network["predictions"] = concatenate([network["predictions_loc"],
                                          network["predictions_conf"]],
                                         axis=2,
                                         name="predictions")
    # model
    model = Model(network["input"], network["predictions"])
    # bbox
    all_bboxes = np.concatenate(bboxes, axis=0).astype("float32")

    return model, all_bboxes
