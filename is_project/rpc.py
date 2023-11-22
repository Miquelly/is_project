import sys
from typing import Union

import cv2
import numpy as np
import numpy.typing as npt
from is_wire.rpc.context import Context
from is_project.detector import Detector
from google.protobuf.json_format import Parse
from is_wire.core import Channel, Status, StatusCode
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.rpc import ServiceProvider, LogInterceptor
from google.protobuf.message import Message as PbMessage
from is_project.conf.options_pb2 import ServiceOptions, ImageOptions


def load_json(filename: str, schema: PbMessage) -> PbMessage:
    with open(file=filename, mode="r", encoding="utf-8") as f:
        proto = Parse(f.read(), schema())
    return proto


def to_np(image: Image) -> npt.NDArray[np.uint8]:
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output


class RPCPeopleDetector(Detector):
    def __init__(self, options: ImageOptions):
        super().__init__(options)

    def infer(self, image: Image, ctx: Context) -> Union[Status, ObjectAnnotations]:
        try:
            array = to_np(image)
            (regions, scores) = super().detect(array)
            return super().to_oa(regions, scores, array)
        except Exception:
            return Status(code=StatusCode.INTERNAL_ERROR)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "options.json"
    options = load_json(
        filename=filename,
        schema=ServiceOptions,
    )

    detection = RPCPeopleDetector(options.images)

    channel = Channel(f"amqp://guest:guest@{options.address}")

    provider = ServiceProvider(channel)
    provider.add_interceptor(LogInterceptor())

    provider.delegate(
        topic="PeopleDetector.Detect",
        function=detection.infer,
        request_type=Image,
        reply_type=ObjectAnnotations,
    )

    provider.run()


if __name__ == "__main__":
    main()
