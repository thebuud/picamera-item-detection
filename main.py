import time
import numpy as np
import threading

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

MODEL = InceptionV3(weights="imagenet")

with open("imagenet-classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load pretrained object detection model (example: YOLO / SSD)
# ...
HAND_DETECTION_RESULT = None

OBJECT_DETECTION_RESULT = {}


def detector_callback(result, unused_output_image: mp.Image, timestamp):
    global HAND_DETECTION_RESULT
    HAND_DETECTION_RESULT = result


def detect_object(frame, threshold: float, top_n: int, result_store: dict = None):
    print("Starting object detection in sub thread")
    # result store is None when running in main thread
    if result_store is None:
        result_store = {}

    resized_frame_rgb = cv2.resize(frame, (299, 299))

    img_array = np.expand_dims(resized_frame_rgb, axis=0)

    img_array = preprocess_input(img_array)

    predictions = MODEL.predict(img_array)

    decoded_predictions = decode_predictions(predictions, top=top_n)[0]

    results: list[tuple[str, float]] = []
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        if score >= threshold:
            results.append((label, score))

    result_store["results"] = results
    print("Done detecting object")

    return result_store


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

    image_detector_thread = None

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

        if HAND_DETECTION_RESULT:
            for idx in range(len(HAND_DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = HAND_DETECTION_RESULT.hand_landmarks[idx]
                handedness = HAND_DETECTION_RESULT.handedness[idx]

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

                if image_detector_thread is None:
                    x1, y1, x2, y2 = text_x, text_y, *pt2

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    translate_to_origin = translation_matrix(-center_x, -center_y)
                    scale = scaling_matrix(1.5, 1.5)
                    translate_back = translation_matrix(center_x, center_y)
                    transformation_matrix = translate_back @ scale @ translate_to_origin

                    # first array is x coord of each point
                    # points are ordered TL, TR, BR, BL
                    # T = Top, B = Bottom, L = Left, R = Right
                    # [T, T, B, B] <- x coord
                    # [L, R, R, L] <- y coord
                    # [1, 1, 1, 1] <- z coord if it exist
                    new_vertices = np.dot(
                        transformation_matrix,
                        np.array(
                            [
                                [x1, x2, x2, x1],
                                [y1, y1, y2, y2],
                                [1, 1, 1, 1],
                            ]
                        ),
                    )
                    scaled_pt1 = (int(new_vertices[0][0]), int(new_vertices[1][0]))
                    scaled_pt2 = (int(new_vertices[0][2]), int(new_vertices[1][2]))

                    # crop image using np slicing since image is stored as a numpy array
                    cropped_frame = current_frame[
                        scaled_pt1[1] : scaled_pt2[1], scaled_pt1[0] : scaled_pt2[0]
                    ]

                    image_detector_thread = threading.Thread(
                        target=detect_object,
                        args=[cropped_frame, 0.80, 3, OBJECT_DETECTION_RESULT],
                    )
                    image_detector_thread.start()

                    cv2.rectangle(
                        current_frame,
                        scaled_pt1,
                        scaled_pt2,
                        (0, 255, 0),
                        2,
                        cv2.LINE_8,
                    )

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

        if image_detector_thread is not None and not image_detector_thread.is_alive():
            print(OBJECT_DETECTION_RESULT["results"])
            image_detector_thread = None

        cv2.imshow("Hand Landmarker", current_frame)

        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cv2.destroyAllWindows()


def scaling_matrix(sx, sy):
    return np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ]
    )


def translation_matrix(tx, ty):
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ]
    )


if __name__ == "__main__":
    main()
