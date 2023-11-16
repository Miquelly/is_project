import argparse
import socket
from typing import Tuple

import imutils
import cv2
import numpy as np
from is_msgs.image_pb2 import ObjectAnnotations, Image
from is_wire.core import Channel, Subscription, Message, ContentType


class StreamChannel(Channel):
    def __init__(
        self, uri: str = "amqp://guest:guest@localhost:5672", exchange: str = "is"
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(self) -> Tuple[Message, int]:
        dropped = 0
        msg = super().consume()
        while True:
            try:
                # will raise an exception when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped)


canal = StreamChannel(f"amqp://guest:guest@10.20.5.2:30000")
assinatura = Subscription(canal, name="test")
assinatura.subscribe(topic=f"PeopleDetector.0.Detection")

messagem, _ = canal.consume_last()
imagem = messagem.unpack(ObjectAnnotations)
print(imagem)
