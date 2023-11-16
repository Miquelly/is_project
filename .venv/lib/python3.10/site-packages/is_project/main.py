import argparse
import socket
from typing import Tuple

import imutils
import cv2
import numpy as np
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

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        messagem, _ = canal.consume_last()
        imagem = messagem.unpack(Image)
        array = to_np(imagem)

        resized = imutils.resize(array, width=min(500, array.shape[1]))
        (regions, scores) = hog.detectMultiScale(
            resized, winStride=(4, 4), padding=(4, 4), scale=1.05
        )

        x_scale = array.shape[1] / resized.shape[1]
        y_scale = array.shape[0] / resized.shape[0]

        for (x, y, w, h), score in zip(regions, scores):
            if score > 0.5:
                x_1 = int(x * x_scale)
                y_1 = int(y * y_scale)

                x_2 = int((x + w) * x_scale)
                y_2 = int((y + h) * y_scale)

                cv2.rectangle(array, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)

                text = f"Score: {score}"
                cv2.putText(
                    array,
                    text,
                    (x_1, y_1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        msg = Message()
        msg.content_type = ContentType.PROTOBUF
        estrutura_recebida = to_image(array)
        msg.pack(estrutura_recebida)

        canal.publish(msg, topic=f"PeopleDetector.0.Rendered")

        # cv2.imshow("Display window", array)
        # k = cv2.waitKey(1)
        # if k == ord("q"):
        # break


# No final do while True, publicar array (independentemente de ter detectado algo e desenhado) como um objeto protobuf Image
# Procurar no is_face_detector como que converte que np.array para Image
# criar uma mensagem de resultado
# mensagem deve ser publicada com tópico f"PeopleDetector.{args.camera}.Rendered"
# abrir google chrome em : http://10.20.5.3:30300/PeopleDetector.2.Rendered

if __name__ == "__main__":
    main()
