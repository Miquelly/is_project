import argparse
import socket
from typing import Tuple

import imutils
import cv2
import numpy as np
from is_msgs.image_pb2 import ObjectAnnotations, Image
from is_wire.core import Channel, Subscription, Message, ContentType


class Detector:
    def __init__(self):
        self.detector = cv2.HOGDescriptor()

        self.x_scale = 0
        self.y_scale = 0

    def detect(self, image):
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        resized = imutils.resize(image, width=min(500, image.shape[1]))

        self.x_scale = image.shape[1] / resized.shape[1]
        self.y_scale = image.shape[0] / resized.shape[0]

        return self.detector.detectMultiScale(
            resized, winStride=(4, 4), padding=(4, 4), scale=1.05
        )

    def to_oa(self, results, score, image):
        resized = imutils.resize(image, width=min(500, image.shape[1]))

        annotations = ObjectAnnotations()
        for results, score in zip(results, score):
            item = annotations.objects.add()

            result_1 = item.region.vertices.add()
            result_1.x = int(results[0] * self.x_scale)
            result_1.y = int(results[1] * self.y_scale)

            result_2 = item.region.vertices.add()
            result_2.x = int((results[0] + results[2]) * self.x_scale)
            result_2.y = int((results[1] + results[3]) * self.y_scale)

            item.label = "human_face"
            item.score = score

        annotations.resolution.width = resized.shape[1]
        annotations.resolution.height = resized.shape[0]
        return annotations

    def draw_detections(self, image, annotations):
        # resized = imutils.resize(image, width=min(500, image.shape[1]))

        for obj in annotations.objects:
            x1 = int(obj.region.vertices[0].x)
            y1 = int(obj.region.vertices[0].y)
            x2 = int(obj.region.vertices[1].x)
            y2 = int(obj.region.vertices[1].y)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            text = f"Score: {obj.score}"
            cv2.putText(
                image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        return image
