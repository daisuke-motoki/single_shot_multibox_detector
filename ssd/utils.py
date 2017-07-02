import numpy as np


def make_bboxes(input_shape, feature_map_shape,
                aspect_ratios, scale):
    """
    """
    map_w = feature_map_shape[1]
    map_h = feature_map_shape[2]
    input_w = input_shape[0]
    input_h = input_shape[1]

    # local box's sizes
    min_size = scale[0]
    box_w = [min_size]
    box_h = [min_size]
    if len(scale) == 2:
        box_w.append(np.sqrt(min_size * scale[1]))
        box_h.append(np.sqrt(min_size * scale[1]))
    for ar in aspect_ratios:
        box_w.append(min_size * np.sqrt(ar))
        box_h.append(min_size / np.sqrt(ar))
    box_w = np.array(box_w)/2/input_w
    box_h = np.array(box_h)/2/input_h

    # feature grids
    step_w = input_w / map_w
    step_h = input_h / map_h
    center_w, center_h = np.mgrid[0:map_w, 0:map_h] + 0.5
    center_w = (center_w * step_w/input_w).reshape(-1, 1)
    center_h = (center_h * step_h/input_h).reshape(-1, 1)

    n_local_box = len(box_w)
    bboxes = np.concatenate((center_w, center_h), axis=1)
    bboxes = np.tile(bboxes, (1, 2 * n_local_box))
    bboxes[:, ::4] -= box_w
    bboxes[:, 1::4] -= box_h
    bboxes[:, 2::4] += box_w
    bboxes[:, 3::4] += box_h
    bboxes = bboxes.reshape(-1, 4)
    bboxes = np.minimum(np.maximum(bboxes, 0.0), 1.0)

    return bboxes


class BoundaryBox:
    """
    """
    def __init__(self, n_classes=0, default_boxes=None,
                 overlap_threshold=0.5, nms_threshold=0.45,
                 max_output_size=400):
        self.n_classes = n_classes
        self.default_boxes = default_boxes
        self.overlap_threshold = overlap_threshold
        self.nms_threshold = nms_threshold

    def encode(self):
        pass

    def decode(self):
        pass

