import pickle
from ssd.models import AVAILABLE_TYPE
from ssd.models import SSD300


class SingleShotMultiBoxDetector:
    """
    """
    ar_presets = dict(
        ssd300=[[2., 1/2.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2., 3., 1/3.],
                [2., 1/2.],
                [2., 1/2.]]
        # ssd300=[[2., 1/2.],
        #         [2., 1/2., 3., 1/3.],
        #         [2., 1/2., 3., 1/3.],
        #         [2., 1/2., 3., 1/3.],
        #         [2., 1/2., 3., 1/3.],
        #         [2., 1/2., 3., 1/3.]]
    )
    scale_presets = dict(
        ssd300=[(30., 60.),
                (60., 111.),
                (111., 162.),
                (162., 213.),
                (213., 264.),
                (264., 315.)]
        # ssd300=[(30.),
        #         (60., 111.),
        #         (111., 162.),
        #         (162., 213.),
        #         (213., 264.),
        #         (264., 315.)]
    )
    default_shapes = dict(
        ssd300=(300, 300, 3)
    )

    def __init__(self, n_classes, input_shape=None, aspect_ratios=None,
                 scales=None, variances=None,
                 model_type="ssd300", base_net="VGG16"):
        """
        """
        self.n_classes = n_classes
        if input_shape:
            self.input_shape = input_shape
        else:
            self.input_shape = self.default_shapes[model_type]
        if aspect_ratios:
            self.aspect_ratios = aspect_ratios
        else:
            self.aspect_ratios = self.ar_presets[model_type]
        if scales:
            self.scales = scales
        else:
            self.scales = self.scale_presets[model_type]
        if variances:
            self.variances = variances
        else:
            self.variances = [0.1, 0.1, 0.2, 0.2]
        self.model_type = model_type
        self.base_net = base_net

        self.model = None
        self.bboxes = None

    def build(self, init_weight="keras_imagenet"):
        """
        """
        # create network
        if self.model_type == "ssd300":
            self.model, priors = SSD300(self.input_shape,
                                        self.n_classes,
                                        self.base_net,
                                        self.aspect_ratios,
                                        self.scales,
                                        self.variances)
            if init_weight is None:
                pass
            elif init_weight == "keras_imagenet":
                from keras.applications import vgg16 as keras_vgg16
                weights_path = keras_vgg16.get_file(
                    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    keras_vgg16.WEIGHTS_PATH_NO_TOP,
                    cache_subdir="models"
                )
                self.model.load_weights(weights_path, by_name=True)
            else:
                raise NameError(
                    "{} is not defined.".format(
                        init_weight
                    )
                )

        else:
            raise NameError(
                "{} is not defined. Please select from {}".format(
                    self.model_type, AVAILABLE_TYPE
                )
            )

        # make boundary box class
        self.bboxes = BoundaryBox(n_classes=self.n_classes,
                                  default_boxes=priors,
                                  overlap_threshold=0.5,
                                  nms_threshold=0.45,
                                  max_output_size=400)

    def train(self):
        """
        """
        pass

    def save_parameters(self, filepath="./param.pkl"):
        """
        """
        params = dict(
            n_classes=self.n_classes,
            input_shape=self.input_shape,
            model_type=self.model_type,
            base_net=self.base_net,
            aspect_ratios=self.aspect_ratios,
            scales=self.scales,
            variances=self.variances
        )
        print("Writing parameters into {}.".format(filepath))
        pickle.dump(params, open(filepath, "wb"))
