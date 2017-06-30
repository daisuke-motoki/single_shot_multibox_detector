from ssd.models import SSD300


if __name__ == "__main__":
    n_classes = 10
    input_shape = (300, 300, 3)

    ssd = SSD300(input_shape, n_classes)
    ssd.build()
    ssd.train()
