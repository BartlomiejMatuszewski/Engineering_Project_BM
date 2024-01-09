# -*- coding: utf-8 -*-
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from KeyPointClassifier import KeyPointClassifier
from GestureNumber import GestureNumber
from get_row_for_model import get_row_for_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def detect_hand_gesture(mp_model_path: str = os.path.join("../frameworks_dnn_models/hand_landmarker.task"),
                        tflite_model_path: str = os.path.join("../trained_dnn_models/tflite_models/model1.tflite"),
                        number_of_hands: int = 1,
                        min_hand_detection_confidence: float = 0.5,
                        min_hand_presence_confidence: float = 0.5,
                        min_tracking_confidence: float = 0.5,
                        camera_id: int = 0,
                        gesture_buffor: int = 10 # defines length of buffer
                        ):
    # we need to specify number of camera from which we take video signal
    cap = cv2.VideoCapture(camera_id)
    DETECTION_RESULT = None
    # buffer for gestures
    RESULTS_LIST: list = []

    # utility function, used for most common gesture in buffer, buffer takes k gesture detection form k last frames
    # gesture most common is chosen as detected
    def most_common(lst: list):
        return max(set(lst), key=lst.count)

    # callback function,
    def save_result(result: vision.HandLandmarkerResult,
                    unused_output_image: mp.Image,
                    timestamp: int
                    ):
        # thanks to 'nonlocal' you can take variable from outer scope function
        nonlocal DETECTION_RESULT

        DETECTION_RESULT = result

    base_options = python.BaseOptions(model_asset_path=mp_model_path)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=number_of_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result # callback function, called after landmarks recognition
    )

    # initializing mediapipe model for landmarks
    handlandmarker: vision.HandLandmarker = vision.HandLandmarker.create_from_options(options)
    # initializing gesture recognizer
    classificator: KeyPointClassifier = KeyPointClassifier(model_path=tflite_model_path)

    while cap.isOpened():
        # reading frame
        success, bgr_image = cap.read()

        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        bgr_image = cv2.flip(bgr_image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run hand landmarker using the model.
        handlandmarker.detect_async(mp_image, time.time_ns() // 1_000_000)

        # detection result may be not none but is empty and program crashes
        if DETECTION_RESULT is not None and len(DETECTION_RESULT.hand_landmarks) > 0:
            # getting x and y parts of coordinates
            row = get_row_for_model(DETECTION_RESULT)["row"]
            # gesture recognizer inference
            my_model_output = classificator(row)

            # if buffer gets full, returning results
            if len(RESULTS_LIST) >= gesture_buffor:

                most_common_gesture = most_common(RESULTS_LIST)
                print()
                print("Gest ręki : " + str(GestureNumber(most_common_gesture)))
                print()
                RESULTS_LIST = []

            else:
                RESULTS_LIST.append(my_model_output)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    handlandmarker.close()
    cap.release()

# if __name__ == '__main__':
#     detect_hand_gesture()
