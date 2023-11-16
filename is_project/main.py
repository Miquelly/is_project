import argparse
import socket
from typing import Tuple

import imutils
import cv2
import numpy as np
from is_project.detector import Detector
from is_msgs.image_pb2 import Image
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


def to_np(image):
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output


def to_image(
    image, encode_format: str = ".jpeg", compression_level: float = 0.8
) -> Image:
    if encode_format == ".jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
    else:
        return Image()
    cimage = cv2.imencode(ext=encode_format, img=image, params=params)
    return Image(data=cimage[1].tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endereco",
        help="Endereco",
        type=str,
        default="10.20.5.2:30000",
    )
    parser.add_argument(
        "--camera",
        help="Camera",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    canal = StreamChannel(f"amqp://guest:guest@{args.endereco}")
    assinatura = Subscription(canal, name="PeopleDetector")
    assinatura.subscribe(topic=f"CameraGateway.{args.camera}.Frame")

    detection = Detector()

    while True:
        messagem, _ = canal.consume_last()
        image = messagem.unpack(Image)
        array = to_np(image)

        (regions, scores) = detection.detect(array)

        results = Message()
        results.content_type = ContentType.PROTOBUF
        estrutura_recebida = detection.to_oa(regions, scores, array)
        results.pack(estrutura_recebida)

        image_detect = detection.draw_detections(array, estrutura_recebida)

        msg = Message()
        msg.content_type = ContentType.PROTOBUF
        estrutura_recebida = to_image(image_detect)
        msg.pack(estrutura_recebida)

        canal.publish(msg, topic=f"PeopleDetector.{args.camera}.Rendered")
        canal.publish(results, topic=f"PeopleDetector.{args.camera}.Detection")


if __name__ == "__main__":
    main()
