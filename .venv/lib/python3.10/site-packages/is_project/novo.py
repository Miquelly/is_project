import argparse

import imutils
import cv2
import numpy as np
from is_msgs.image_pb2 import Image
from is_wire.core import Channel, Subscription, Message, ContentType


def to_np(image):
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output


canal = Channel("amqp://guest:guest@10.20.5.2:30000")
assinatura = Subscription(canal)
assinatura.subscribe(topic="PeopleDetector.0.Rendered")

msg = canal.consume()
imagem = msg.unpack(Image)
array = to_np(imagem)

print(array)
