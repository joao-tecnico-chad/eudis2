"""Tests for YOLO output decoding and NMS."""

import numpy as np

from guardian.utils.decode import Detection, decode_yolov6, decode_yolov8, nms


class TestNMS:
    def test_empty(self):
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        assert nms(boxes, scores, 0.5) == []

    def test_single_box(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9])
        assert nms(boxes, scores, 0.5) == [0]

    def test_overlapping_boxes_suppressed(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],  # high overlap with first
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        kept = nms(boxes, scores, 0.5)
        assert len(kept) == 1
        assert kept[0] == 0  # higher score kept

    def test_non_overlapping_boxes_kept(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [200, 200, 250, 250],  # no overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        kept = nms(boxes, scores, 0.5)
        assert len(kept) == 2


class TestDecodeYOLOv6:
    def test_basic_decode(self):
        # Single detection: cx=208, cy=208, w=83, h=83, obj=0.9, cls0=0.95
        # Coordinates in pixel space (as OAK blob outputs)
        data = np.array([[208, 208, 83, 83, 0.9, 0.95]], dtype=np.float32)
        dets = decode_yolov6(data, img_size=416, num_classes=1, conf_thresh=0.3, iou_thresh=0.5)
        assert len(dets) == 1
        d = dets[0]
        assert d.class_id == 0
        assert d.confidence > 0.8
        # cx=208, w=83 -> x1=208-41=167, x2=208+41=249
        assert 160 <= d.x1 <= 170
        assert 245 <= d.x2 <= 255

    def test_low_confidence_filtered(self):
        data = np.array([[208, 208, 83, 83, 0.1, 0.1]], dtype=np.float32)
        dets = decode_yolov6(data, img_size=416, num_classes=1, conf_thresh=0.3, iou_thresh=0.5)
        assert len(dets) == 0


class TestDecodeYOLOv8:
    def test_basic_decode(self):
        # YOLOv8 output: [1, 5, N] where rows are [cx, cy, w, h, score]
        # Single detection at center, 100px wide
        output = np.zeros((1, 5, 3), dtype=np.float32)
        output[0, 0, 0] = 320  # cx
        output[0, 1, 0] = 240  # cy
        output[0, 2, 0] = 100  # w
        output[0, 3, 0] = 80   # h
        output[0, 4, 0] = 0.9  # score
        # Two low-confidence detections
        output[0, 4, 1] = 0.1
        output[0, 4, 2] = 0.1

        dets = decode_yolov8(output, scale=1.0, conf_thresh=0.3, iou_thresh=0.5)
        assert len(dets) == 1
        d = dets[0]
        assert d.x1 == 270  # 320 - 50
        assert d.y1 == 200  # 240 - 40
        assert d.x2 == 370  # 320 + 50
        assert d.y2 == 280  # 240 + 40

    def test_scale_applied(self):
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, 0, 0] = 320
        output[0, 1, 0] = 240
        output[0, 2, 0] = 100
        output[0, 3, 0] = 80
        output[0, 4, 0] = 0.9

        dets = decode_yolov8(output, scale=0.5, conf_thresh=0.3, iou_thresh=0.5)
        assert len(dets) == 1
        d = dets[0]
        # Coordinates divided by scale
        assert d.x1 == 540  # (320-50)/0.5
        assert d.x2 == 740  # (320+50)/0.5
