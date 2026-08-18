"""Standalone SCRFD face/landmark detector on ONNX Runtime.

Minimal reimplementation of InsightFace's SCRFD inference (det_500m.onnx,
buffalo_sc pack) so we don't need the insightface package. Outputs per face:
bbox (x1,y1,x2,y2), det_score, and 5 landmarks (left eye, right eye, nose,
left mouth, right mouth) — the same contract as the production detector.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float = 0.4) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class SCRFD:
    """SCRFD detector matching production settings (det_size 160x160, thresh 0.5)."""

    def __init__(self, model_path: str | Path, det_size: tuple[int, int] = (160, 160),
                 det_thresh: float = 0.5):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # silence benign VerifyOutputSizes warnings
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self._strides = (8, 16, 32)
        self._num_anchors = 2
        self._center_cache: dict[tuple, np.ndarray] = {}

    def _forward(self, img: np.ndarray):
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 128, self.det_size, (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})
        scores_list, bboxes_list, kps_list = [], [], []
        input_h, input_w = blob.shape[2], blob.shape[3]
        fmc = 3  # feature map count
        for idx, stride in enumerate(self._strides):
            scores = outs[idx]
            bbox_preds = outs[idx + fmc] * stride
            kps_preds = outs[idx + fmc * 2] * stride
            h, w = input_h // stride, input_w // stride
            key = (h, w, stride)
            if key in self._center_cache:
                centers = self._center_cache[key]
            else:
                centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                centers = np.stack([centers] * self._num_anchors, axis=1).reshape(-1, 2)
                self._center_cache[key] = centers
            mask = scores.ravel() >= self.det_thresh
            bboxes = _distance2bbox(centers, bbox_preds)
            kps = _distance2kps(centers, kps_preds)
            scores_list.append(scores.reshape(-1)[mask])
            bboxes_list.append(bboxes[mask])
            kps_list.append(kps[mask])
        return (np.concatenate(scores_list), np.concatenate(bboxes_list),
                np.concatenate(kps_list))

    def detect(self, bgr: np.ndarray):
        """Detect the highest-scoring face.

        Returns (bbox[4], score, kps[5,2]) in original image coords, or None.
        """
        im_h, im_w = bgr.shape[:2]
        model_w, model_h = self.det_size
        scale = min(model_w / im_w, model_h / im_h)
        new_w, new_h = int(im_w * scale), int(im_h * scale)
        resized = cv2.resize(bgr, (new_w, new_h))
        padded = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        scores, bboxes, kps = self._forward(padded)
        if scores.size == 0:
            return None
        dets = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, 0.4)
        dets, kps = dets[keep], kps[keep]
        best = int(np.argmax(dets[:, 4]))
        bbox = dets[best, :4] / scale
        score = float(dets[best, 4])
        landmarks = kps[best].reshape(5, 2) / scale
        return bbox, score, landmarks
