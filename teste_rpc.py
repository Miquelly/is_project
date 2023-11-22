import cv2
import socket

import numpy as np
import numpy.typing as npt
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.core import Channel, Message, Subscription


def to_image(
    image: npt.NDArray[np.uint8],
    encode_format: str = ".jpeg",
    compression_level: float = 0.8,
) -> Image:
    if encode_format == ".jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
    else:
        return Image()
    cimage = cv2.imencode(ext=encode_format, img=image, params=params)
    return Image(data=cimage[1].tobytes())


def draw_detections(
    image: npt.NDArray[np.uint8], annotations: ObjectAnnotations
) -> npt.NDArray[np.uint8]:
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


def main():
    image = cv2.imread("test2.jpg")
    channel = Channel("amqp://guest:guest@10.20.5.2:30000")
    subscription = Subscription(channel)

    request = Message(
        content=to_image(image=image),
        reply_to=subscription,
    )
    channel.publish(request, topic="PeopleDetector.Detect")
    try:
        reply = channel.consume(timeout=5.0)
        response = reply.unpack(ObjectAnnotations)
        print("RPC Status:", reply.status, "\nResponse:", response)
    except socket.timeout:
        print("No reply :(")

    image_detection = draw_detections(image, response)
    cv2.imshow("Imagem", image_detection)
    # Aguardar por um evento de teclado (0 indica esperar indefinidamente)
    cv2.waitKey(0)

    # Fechar a janela
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
