from ssd.ssd import SingleShotMultiBoxDetector


if __name__ == "__main__":
    n_classes = 10
    input_shape = (300, 300, 3)

    ssd = SingleShotMultiBoxDetector(model_type="ssd300",
                                     base_net="VGG16",
                                     n_classes=n_classes)
    ssd.build()
    ssd.save_parameters("ssd300_params.pkl")
