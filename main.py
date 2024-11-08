import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from picamera2 import Picamera2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection module
hands = mp_hands.Hands()

# Load pretrained object detection model (example: YOLO / SSD)
# ...
DETECTION_RESULT = None


def detector_callback(result, unused_output_image: mp.Image, timestamp):
    global DETECTION_RESULT
    print("Hand detected")
    DETECTION_RESULT = result


def main():
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.open(-1)

    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=detector_callback,
    )

    detector = vision.HandLandmarker.create_from_options(options)
    print(cap.isOpened())

    while cap.isOpened():
        success, image = cap.read()
        alt_success, alt_image = cap.retrieve()

        if not success and not alt_success:
            print("failed")
            break

        image = image or alt_image

        image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

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
                    image,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                height, width, _ = image.shape
                x_coords = [landmark.x for landmark in hand_landmarks]
                y_coords = [landmark.y for landmark in hand_landmarks]

                text_x = int(min(x_coords) * width)
                text_y = int(min(y_coords) * height) - 10

                cv2.putText(
                    image,
                    f"{handedness[0].category_name}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (88, 205, 54),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Hand Landmarker", image)

        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def alt_main():
    cam = Picamera2()
    cam.configure(
        cam.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )
    )
    cam.start()

    while True:
        frame = cam.capture_array()

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for hand detection
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            print("hand detected")

            # Extract hand ROI
            # ...

            # Apply object detection within hand ROI
            # ...

        cv2.imshow("Hand Object Detection", frame)
        if cv2.waitKey(10) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # alt_main()
    main()
