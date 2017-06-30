

class BoundaryBoxes:
    """
    """
    def __init__(self, n_classes=0, prior_boxes=None,
                 overlap_threshold=0.5, nms_threshold=0.45,
                 max_output_size=400):
        self.n_classes = n_class
        self.prior_boxes = prior_boxes
        self.overlap_threshold = overlap_threshold
        self.nms_threshold = nms_threshold

    def encode(self):
        pass

    def decode(self):
        pass

    def make_boxes(self):
        pass
