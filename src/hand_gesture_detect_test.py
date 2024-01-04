import os.path
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from src import GestureNumber, KeyPointClassifier, get_row_for_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None


def run_inference(model_path: str,
                  number_of_hands: int,
                  min_hand_detection_confidence: float,
                  min_hand_presence_confidence: float,
                  min_tracking_confidence: float,
                  camera_id: int,
                  frame_width: int,
                  frame_height: int
                  ):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    row_size = 50
    left_margin = 23

    # Visualization parameters
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(result: vision.HandLandmarkerResult,
                    unused_output_image: mp.Image,
                    timestamp: int
                    ):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=number_of_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result
    )

    path1 = os.path.join("../trained_models/tflite_models/model1.tflite")
    path2 = os.path.join("../trained_models/tflite_models/model2.tflite")

    detector: vision.HandLandmarker = vision.HandLandmarker.create_from_options(options)
    classificator = KeyPointClassifier.KeyPointClassifier(model_path=path1)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run hand landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        # Landmark visualization parameters.
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        if DETECTION_RESULT:
            # Draw landmarks and indicate handedness.
            for idx in range(len(DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]
                handedness = DETECTION_RESULT.handedness[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in hand_landmarks
                ])

                mp_drawing.draw_landmarks(
                    current_frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = current_frame.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN

                # Draw handedness (left or right hand) on the image.
                cv2.putText(current_frame, f"{handedness[0].category_name}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS,
                            cv2.LINE_AA)

                row = get_row_for_model.get_row_for_model(DETECTION_RESULT)["row"]
                my_model_output = classificator(row)
                print()
                print(GestureNumber.GestureNumber(my_model_output))
                print()

        cv2.imshow('hand_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    model_path = os.path.join("../frameworks_models/hand_landmarker.task")
    number_of_hands: int = 1

    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 960

    run_inference(model_path, number_of_hands, min_hand_detection_confidence, min_hand_presence_confidence,
                  min_tracking_confidence, camera_id, frame_width, frame_height)


if __name__ == '__main__':
    main()
