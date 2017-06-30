from abc import abstractmethod
from collections import OrderedDict
from keras.layers import Input, Conv2D, MaxPool2D, AtrousConv2D
from keras.layers import Flatten, Reshape, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from ssd.layers import L2Normalization


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


def SSD300(input_shape, n_classes):
    """
    """
    n_classes += 1  # add background class
    network = OrderedDict()
    network["input"] = Input(shape=input_shape)

    # base network -------------------------------------------------------
    _build_vgg16_basenet(network)

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
    # extra feature layer --------------------------------------------

    # classifier -----------------------------------------------------
    # block4 scale classifier
    network["block4_norm"] = L2Normalization(20,
                                             name="block4_norm"
                                             )(network["block4_conv3"])

    list_n_boxes = [4, 6, 6, 6, 6, 6]
    list_aspect_ratios = [[2],
                          [2, 3],
                          [2, 3],
                          [2, 3],
                          [2, 3],
                          [2, 3]]
    classifier_layers = ["block4_norm", "block7_conv", "block8_conv2",
                         "block9_conv2", "block10_conv2", "block11_conv2"]
    for n_boxes, layer_name in zip(list_n_boxes, classifier_layers):
        network[layer_name+"_loc"] = Conv2D(n_boxes * 4, (3, 3),
                                            padding="same",
                                            name=layer_name+"_loc"
                                            )(network[layer_name])
        network[layer_name+"_loc_flat"] = Flatten(
            name=layer_name+"_loc_flat")(network[layer_name+"_loc"])
        network[layer_name+"_conf"] = Conv2D(n_boxes * n_classes, (3, 3),
                                             padding="same",
                                             name=layer_name+"_conf"
                                             )(network[layer_name])
        network[layer_name+"_conf_flat"] = Flatten(
            name=layer_name+"_conf_flat")(network[layer_name+"_conf"])
    # classifier -----------------------------------------------------

    # collect predictions
    list_loc_layers = list()
    list_conf_layers = list()
    for layer_name in classifier_layers:
        list_loc_layers.append(network[layer_name+"_loc_flat"])
        list_conf_layers.append(network[layer_name+"_conf_flat"])
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

    return model


def SSD512():
    """
    """
    pass
