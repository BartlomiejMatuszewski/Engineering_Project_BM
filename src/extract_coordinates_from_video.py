import os
import time

import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2

from GestureNumber import GestureNumber


def extract_coordinates_from_video(video_path,
                                   category: GestureNumber = GestureNumber.POINT_UP,
                                   model_path=os.path.join("../models/hand_landmarker.task"),
                                   output_dir=os.path.join("../csv_data"),
                                   output_name="default_name"
                                   ):
    # for clarity
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    FPS = 0
    START_TIME = time.time()

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Label box parameters
    label_text_color = (0, 0, 0)  # red
    label_background_color = (255, 255, 255)  # white
    label_font_size = 1
    label_thickness = 2
    label_padding_width = 100  # pixels

    recognition_frame = None
    recognition_result_list = []

    def row_for_csv(row_lst, gesture_cat):
        HEADERS = ['WRIST_x', 'WRIST_y', 'THUMB_CMC_x', 'THUMB_CMC_y', 'THUMB_MCP_x', 'THUMB_MCP_y', 'THUMB_IP_x',
                   'THUMB_IP_y',
                   'THUMB_TIP_x', 'THUMB_TIP_y', 'INDEX_FINGER_MCP_x', 'INDEX_FINGER_MCP_y', 'INDEX_FINGER_PIP_x',
                   'INDEX_FINGER_PIP_y', 'INDEX_FINGER_DIP_x', 'INDEX_FINGER_DIP_y', 'INDEX_FINGER_TIP_x',
                   'INDEX_FINGER_TIP_y',
                   'MIDDLE_FINGER_MCP_x', 'MIDDLE_FINGER_MCP_y', 'MIDDLE_FINGER_PIP_x', 'MIDDLE_FINGER_PIP_y',
                   'MIDDLE_FINGER_DIP_x', 'MIDDLE_FINGER_DIP_y', 'MIDDLE_FINGER_TIP_x', 'MIDDLE_FINGER_TIP_y',
                   'RING_FINGER_MCP_x', 'RING_FINGER_MCP_y', 'RING_FINGER_PIP_x', 'RING_FINGER_PIP_y',
                   'RING_FINGER_DIP_x',
                   'RING_FINGER_DIP_y', 'RING_FINGER_TIP_x', 'RING_FINGER_TIP_y', 'PINKY_MCP_x', 'PINKY_MCP_y',
                   'PINKY_PIP_x',
                   'PINKY_PIP_y', 'PINKY_DIP_x', 'PINKY_DIP_y', 'PINKY_TIP_x', 'PINKY_TIP_y', "GESTURE"]

        file_name = output_name + ".csv"
        file_location = os.path.join(output_dir, file_name)
        df = pd.DataFrame(columns=HEADERS)

        for row in landmarks_dataset:
            df.loc[len(df)] = row + [gesture_cat.value]

        df.to_csv(file_location)

    base_opts = BaseOptions(model_path)

    landmarker_opts = HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionRunningMode.VIDEO
    )

    landmarker = HandLandmarker.create_from_options(landmarker_opts)

    landmarks_dataset = []

    capture = cv2.VideoCapture(video_path)

    COUNTER = 0

    while capture.isOpened():
        # fps: float = capture.get(cv2.CAP_PROP_FPS)
        # timestamp: float = capture.get(cv2.CAP_PROP_POS_MSEC)
        # timestamp = int(timestamp)

        success, frame = capture.read()

        if not success:
            # sys.exit('ERROR: Unable to read from video. Please verify further...')
            print("ERROR: Unable to read from video. Please verify further...")
            break

        image = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # print(timestamp)

        result = landmarker.detect_for_video(mp_image, time.time_ns() // 1_000_000)

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        recognition_result_list.append(result)
        COUNTER += 1
        print(COUNTER)

        current_frame = image

        if recognition_result_list:

            for hand_landmarks in recognition_result_list[0].hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                    y=landmark.y,
                                                    z=landmark.z) for landmark in hand_landmarks
                ])

                mp_drawing.draw_landmarks(
                    current_frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                landmarks_dataset.append([])

                for landmark in hand_landmarks:
                    landmarks_dataset[-1].append(landmark.x)
                    landmarks_dataset[-1].append(landmark.y)

        # # Expand the frame to show the labels.
        current_frame = cv2.copyMakeBorder(current_frame, 0, label_padding_width,
                                           0, 0,
                                           cv2.BORDER_CONSTANT, None,
                                           label_background_color)

        recognition_frame = current_frame
        # FIXME: recognition_result_list = [] , chyba
        recognition_result_list.clear()

        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)

        if cv2.waitKey(1) == 27:
            break

    row_for_csv(landmarks_dataset, category)
    capture.release()
    landmarker.close()
    cv2.destroyAllWindows()
