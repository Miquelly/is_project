import cv2
import imutils
import numpy as np
import numpy.typing as npt
from typing import List, Sequence, Tuple
from is_msgs.image_pb2 import ObjectAnnotations


class Detector:
    def __init__(self, images):
        self.detector = cv2.HOGDescriptor()

        self.x_scale = 0
        self.y_scale = 0

        self.width_min = images.width_min
        self.scale = images.scale
        self.winStride = (images.tuple_winStride[0], images.tuple_winStride[1])
        self.padding = (images.tuple_padding[0], images.tuple_padding[1])

    def detect(self, image: npt.NDArray[np.uint8]) -> Tuple[Sequence, Sequence[float]]:
        # def detect(self, image: npt.NDArray[np.uint8]):
        """
        Detect people in an input image using a Histogram of Oriented Gradients (HOG) detector.

        Parameters:
        -----------
        image : np.ndarray
            The input image in the form of a NumPy array.

        Returns:
        --------
        List[Tuple[int, int, int, int]]
            A list of tuples representing the detected bounding boxes of people.
            Each tuple contains (x, y, width, height).

        Notes:
        ------
        - The detection is based on a pre-trained HOG detector.

        Examples:
        ---------
        >>> detector_instance = YourDetectorClass()  # Replace YourDetectorClass with the actual name of your detector class
        >>> image_array = cv2.imread("path/to/your/image.jpg")
        >>> detected_people = detector_instance.detect(image_array)
        """
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        resized = imutils.resize(image, width=min(self.width_min, image.shape[1]))

        self.x_scale = image.shape[1] / resized.shape[1]
        self.y_scale = image.shape[0] / resized.shape[0]

        return self.detector.detectMultiScale(
            resized, winStride=self.winStride, padding=self.padding, scale=self.scale
        )

    def to_oa(
        self, results: Sequence, score: Sequence[float], image: npt.NDArray[np.uint8]
    ) -> ObjectAnnotations:
        """
        Convert detection results and scores to ObjectAnnotations
        in the Open Annotation (OA) format.

        Parameters:
        -----------
        results : List[Tuple[int, int, int, int]]
            A list of tuples representing detected bounding boxes.
            Each tuple contains (x, y, width, height).

        score : List[float]
            A list of confidence scores corresponding to each detection.

        image : np.ndarray
            The input image in the form of a NumPy array.

        Returns:
        --------
        ObjectAnnotations
            An ObjectAnnotations instance containing the converted
            detection results.

        Notes:
        ------
        - The conversion is based on the specified scaling factors
        (self.x_scale and self.y_scale).

        Examples:
        ---------
        >>> converter_instance = YourConverterClass()
        >>> detected_results = [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
        >>> confidence_scores = [score1, score2, ...]
        >>> image_array = cv2.imread("path/to/your/image.jpg")
        >>> oa_annotations = converter_instance.to_oa(
            detected_results, confidence_scores, image_array)
        """
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

        annotations.resolution.width = image.shape[1]
        annotations.resolution.height = image.shape[0]
        return annotations

    def draw_detections(
        self, image: npt.NDArray[np.uint8], annotations: ObjectAnnotations
    ) -> npt.NDArray[np.uint8]:
        """
        Draw bounding boxes and scores on the input image based on ObjectAnnotations.

        Parameters:
        -----------
        image : np.ndarray
            The input image in the form of a NumPy array.

        annotations : ObjectAnnotations
            An ObjectAnnotations instance containing detection information.

        Returns:
        --------
        np.ndarray
            The input image with bounding boxes and scores drawn.

        Examples:
        ---------
        >>> drawer_instance = YourDrawerClass()
        >>> image_array = cv2.imread("path/to/your/image.jpg")
        >>> detection_annotations = YourDetectionAnnotations
        >>> drawn_image = drawer_instance.draw_detections
        (image_array, detection_annotations)
        """
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
