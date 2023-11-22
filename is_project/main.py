import sys
import socket
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
from is_msgs.image_pb2 import Image
from is_project.detector import Detector
from google.protobuf.json_format import Parse
from is_project.conf.options_pb2 import ServiceOptions
from is_wire.core import Channel, Message, Subscription
from google.protobuf.message import Message as PbMessage


class StreamChannel(Channel):
    def __init__(
        self, uri: str = "amqp://guest:guest@localhost:5672", exchange: str = "is"
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)

    """
    A class representing a streaming channel for consuming messages.

    Parameters:
    -----------
    uri : str, optional
        The URI for connecting to the message broker. 
        Defaults to "amqp://guest:guest@localhost:5672".

    exchange : str, optional
        The exchange to bind the channel to. Defaults to "is".

    Methods:
    --------
    consume_last() -> Tuple[Message, int]:
        Consume the last available message from the channel.

    Returns a tuple containing the consumed message and 
    the number of dropped messages.

    Examples:
    ---------
    >>> stream_channel = StreamChannel(uri="amqp://guest:guest@localhost:5672", exchange="is")
    >>> message, dropped_count = stream_channel.consume_last()
    """

    def consume_last(self) -> Tuple[Message, int]:
        """
        Consume the last available message from the channel.

        Returns:
        --------
        Tuple[Message, int]
            A tuple containing the consumed message and the number
            of dropped messages.
        """
        dropped = 0
        msg = super().consume()
        while True:
            try:
                # will raise an exception when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped)


def to_np(image: Image) -> npt.NDArray[np.uint8]:
    """
    Convert a Protocol Buffer Image message to a NumPy array.

    Parameters:
    -----------
    image : Image
        A Protocol Buffer Image message containing the encoded image.

    Returns:
    --------
    np.ndarray
        The NumPy array representing the decoded image.

    Examples:
    ---------
    >>> image_proto = SomeImageMessage
    >>> numpy_image = to_np(image_proto)
    """
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output


def to_image(
    image: npt.NDArray[np.uint8],
    encode_format: str = ".jpeg",
    compression_level: float = 0.8,
) -> Image:
    """
    Convert a NumPy array representing an image to a Protocol Buffer Image message.

    Parameters:
    -----------
    image : np.ndarray
        The NumPy array representing the image.

    encode_format : str, optional
        The encoding format for the image. Defaults to ".jpeg".
        Supported formats: ".jpeg", ".png".

    compression_level : float, optional
        Compression level for the encoded image. Applicable for JPEG and PNG formats.
        For JPEG, the compression level ranges from 0.0 to 1.0
        (higher values mean higher quality).
        For PNG, the compression level ranges from 0.0 to 1.0
        (higher values mean higher compression).
        Defaults to 0.8.

    Returns:
    --------
    Image
        A Protocol Buffer Image message containing the encoded image.

    Examples:
    ---------
    >>> image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> encoded_image = to_image(image_array, encode_format=".jpeg",
    compression_level=0.8)
    """
    if encode_format == ".jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
    else:
        return Image()
    cimage = cv2.imencode(ext=encode_format, img=image, params=params)
    return Image(data=cimage[1].tobytes())


def load_json(filename: str, schema: PbMessage) -> PbMessage:
    """
    Load data from a JSON file and parse it into a Protocol Buffer message.

    Parameters:
    -----------
    filename : str
        The path to the JSON file to be loaded.

    schema : PbMessage
        An instance of the Protocol Buffer message schema that will be used for parsing.

    Returns:
    --------
    PbMessage
        A parsed instance of the Protocol Buffer message.

    Examples:
    ---------
    >>> schema = MyProtoMessage
    >>> loaded_message = load_json("data.json", schema)
    """
    with open(file=filename, mode="r", encoding="utf-8") as f:
        proto = Parse(f.read(), schema())
    return proto


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "options.json"
    options = load_json(
        filename=filename,
        schema=ServiceOptions,
    )

    canal = StreamChannel(f"amqp://guest:guest@{options.address}")
    assinatura = Subscription(canal, name="PeopleDetector")
    for camera in options.cameras:
        assinatura.subscribe(topic=f"CameraGateway.{camera}.Frame")

    detection = Detector(options.images)

    while True:
        messagem, _ = canal.consume_last()
        camera_id = messagem.topic.split(".")[1]
        image = messagem.unpack(Image)
        array = to_np(image)

        (regions, scores) = detection.detect(array)

        results = Message()
        estrutura_recebida = detection.to_oa(regions, scores, array)
        results.pack(estrutura_recebida)

        image_detect = detection.draw_detections(array, estrutura_recebida)

        msg = Message()
        estrutura_recebida = to_image(image_detect)
        msg.pack(estrutura_recebida)

        canal.publish(msg, topic=f"PeopleDetector.{camera_id}.Rendered")
        canal.publish(results, topic=f"PeopleDetector.{camera_id}.Detection")


if __name__ == "__main__":
    main()
