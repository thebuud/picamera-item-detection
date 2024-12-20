import time
import numpy as np
import queue
import threading

import pyttsx3

import cv2
import mediapipe as mp
import tensorflow as tf  # noqa

from keras._tf_keras.keras.applications.resnet_v2 import (
    ResNet152V2,
    decode_predictions,
    preprocess_input,
)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2  # noqa

from picamera2 import Picamera2

# Initialize py text to speach engine queue
text_to_speech_queue = queue.Queue()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection module
hands = mp_hands.Hands()

MODEL = ResNet152V2()

print("Warming up Prediction model")
dummy_input = tf.zeros((1, *MODEL.input_shape[1:]))
MODEL.predict(dummy_input)
print("Warm")

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

    resized_frame_rgb = cv2.resize(frame, MODEL.input_shape[1:3])

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


def text_to_speech_external_loop():
    engine = pyttsx3.init()
    engine.setProperty("rate", engine.getProperty("rate") - 50)
    while True:
        try:
            text = text_to_speech_queue.get()
            engine.say(text)
            engine.runAndWait()
            time.sleep(0.25)
        except queue.Empty:
            time.sleep(0.1)


def main():
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

    text_to_speach_thread = threading.Thread(
        target=text_to_speech_external_loop, args=[]
    )
    text_to_speach_thread.start()

    last_object_recognized = [None, 0]

    frame_count = 0
    while True:
        image = cam.capture_array()
        frame_count += 1

        # artificially reduce fps
        if not frame_count % 4 == 0:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        current_frame = rgb_image

        if HAND_DETECTION_RESULT:
            for idx in range(len(HAND_DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = HAND_DETECTION_RESULT.hand_landmarks[idx]
                handedness = HAND_DETECTION_RESULT.handedness[idx]

                height, width, _ = current_frame.shape
                x_coords = [landmark.x for landmark in hand_landmarks]
                y_coords = [landmark.y for landmark in hand_landmarks]

                # x_coords/y_coords are between 0 and 1. multiplying by width and height scales the relative coords to actual coords ex. min: 0.5 width: 640 coord = 320
                text_x = int(min(x_coords) * width)
                text_y = int(min(y_coords) * height)

                pt2 = (int(max(x_coords) * width), int(max(y_coords) * height))

                cropped_frame = current_frame.copy()

                if image_detector_thread is None:
                    image_detector_thread = threading.Thread(
                        target=detect_object,
                        args=[cropped_frame, 0.85, 10, OBJECT_DETECTION_RESULT],
                    )
                    image_detector_thread.start()
                    cv2.imshow("Detection Image", cropped_frame)

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
            if OBJECT_DETECTION_RESULT["results"]:
                result = OBJECT_DETECTION_RESULT["results"][0]
                print(result)
                if result[0] == last_object_recognized[0]:
                    last_object_recognized[1] += 1
                else:
                    last_object_recognized[0] = result[0]
                    last_object_recognized[1] = 1

                if last_object_recognized[1] == 3:
                    label = (
                        f"an {last_object_recognized[0]}"
                        if last_object_recognized[0][0].upper()
                        in ["A", "E", "I", "O", "U"]
                        else f"a {last_object_recognized[0]}"
                    )
                    text_to_speech_queue.put(
                        f"You have {label} in your hand. That's awesome!"
                    )

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


def normalize_scaled_point(
    x: int, y: int, original_width: int, original_height: int
) -> tuple[int, int]:
    if x < 0:
        x = 0
    elif x > original_width:
        x = original_width

    if y < 0:
        y = 0
    elif y > original_height:
        y = original_height
    return (x, y)


if __name__ == "__main__":
    main()
