import time
import numpy as np

import cv2
import mediapipe as mp
import tensorflow as tf  # noqa
from keras._tf_keras.keras.applications.inception_v3 import (
    InceptionV3,
    decode_predictions,  # noqa: F401
    preprocess_input,  # noqa: F401
)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from picamera2 import Picamera2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection module
hands = mp_hands.Hands()

# net = cv2.dnn.readNetFromTensorflow("./saved_model_1.pb")

with open("imagenet-classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load pretrained object detection model (example: YOLO / SSD)
# ...
DETECTION_RESULT = None


def detector_callback(result, unused_output_image: mp.Image, timestamp):
    global DETECTION_RESULT
    if result.hand_landmarks:
        print(
            f"Hand detected: processed in {(time.time_ns() // 1_000_000) - timestamp}ms"
        )
    DETECTION_RESULT = result


def main():
    # cap = cv2.VideoCapture(-1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # cap.open(-1)
    cam = Picamera2()  # type: ignore # noqa

    cam.configure(
        cam.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )
    )
    cam.start()

    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=detector_callback,
    )

    detector = vision.HandLandmarker.create_from_options(options)

    frame_count = 0
    while True:
        image = cam.capture_array()
        frame_count += 1

        # artificially reduce fps
        if not frame_count % 4 == 0:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = cv2.flip(image, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        current_frame = rgb_image

        if DETECTION_RESULT:
            for idx in range(len(DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]
                handedness = DETECTION_RESULT.handedness[idx]

                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in hand_landmarks
                    ]
                )

                mp_drawing.draw_landmarks(
                    current_frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                height, width, _ = current_frame.shape
                x_coords = [landmark.x for landmark in hand_landmarks]
                y_coords = [landmark.y for landmark in hand_landmarks]

                # x_coords/y_coords are between 0 and 1. multiplying by width and height scales the relative coords to actual coords ex. min: 0.5 width: 640 coord = 320
                text_x = int(min(x_coords) * width)
                text_y = int(min(y_coords) * height)

                pt2 = (int(max(x_coords) * width), int(max(y_coords) * height))

                cv2.rectangle(
                    current_frame, (text_x, text_y), pt2, (0, 0, 255), 2, cv2.LINE_8
                )
                cv2.putText(
                    current_frame,
                    f"{handedness[0].category_name}",
                    (text_x, text_y - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (88, 205, 54),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Hand Landmarker", current_frame)

        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cv2.destroyAllWindows()


def alt_main():
    model = InceptionV3(weights="imagenet")
    cam = Picamera2()  # type: ignore # noqa
    cam.configure(
        cam.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )
    )
    cam.start()

    frame = cam.capture_array()

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame_rgb = cv2.resize(frame_rgb, (299, 299))

    img_array = np.expand_dims(resized_frame_rgb, axis=0)

    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    decoded_predictions = decode_predictions(predictions, top=5)[0]

    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}: {label} ({score * 100:.2f}%)")

    cv2.imshow("Hand Object Detection", frame_rgb)
    cv2.waitKey(10)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    alt_main()
    # main()
