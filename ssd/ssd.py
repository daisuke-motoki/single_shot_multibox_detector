import pickle
import keras
from ssd.models import AVAILABLE_TYPE
from ssd.models import SSD300
from ssd.losses import MultiBoxLoss
from ssd.utils import BoundaryBox


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
                 overlap_threshold=0.5, nms_threshold=0.45,
                 max_output_size=400,
                 model_type="ssd300", base_net="vgg16"):
        """
        """
        self.n_classes = 1 + n_classes  # add background class
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
        self.overlap_threshold = overlap_threshold
        self.nms_threshold = nms_threshold
        self.max_output_size = max_output_size
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
                                        self.scales)
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
                self.model.load_weights(init_weight, by_name=True)

        else:
            raise NameError(
                "{} is not defined. Please select from {}".format(
                    self.model_type, AVAILABLE_TYPE
                )
            )

        # make boundary box class
        self.bboxes = BoundaryBox(n_classes=self.n_classes,
                                  default_boxes=priors,
                                  variances=self.variances,
                                  overlap_threshold=self.overlap_threshold,
                                  nms_threshold=self.nms_threshold,
                                  max_output_size=self.max_output_size)

    def train_by_generator(self, gen, epoch=30, neg_pos_ratio=3.0,
                           learning_rate=1e-3, freeze=None, checkpoints=None):
        """
        """
        # set freeze layers
        if freeze is None:
            freeze = list()

        for L in self.model.layers:
            if L.name in freeze:
                L.trainable = False

        # train setup
        callbacks = list()
        if checkpoints:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoints,
                    verbose=1,
                    save_weights_only=True
                ),
            )

        # def schedule(epoch, decay=0.9):
        #     return base_lr * decay**(epoch)
        # callbacks.append(keras.callbacks.LearningRateScheduler(schedule))
        # optim = keras.optimizers.Adam(lr=learning_rate)
        optim = keras.optimizers.SGD(
            lr=learning_rate, momentum=0.9, decay=0.0005, nesterov=True
        )
        self.model.compile(
            optimizer=optim,
            loss=MultiBoxLoss(
                self.n_classes,
                neg_pos_ratio=neg_pos_ratio
            ).compute_loss
        )
        history = self.model.fit_generator(
            gen.generate(True),
            int(gen.train_batches/gen.batch_size),
            epochs=epoch,
            verbose=1,
            callbacks=callbacks,
            validation_data=gen.generate(False),
            validation_steps=int(gen.val_batches/gen.batch_size),
            workers=1
        )

        return history

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
