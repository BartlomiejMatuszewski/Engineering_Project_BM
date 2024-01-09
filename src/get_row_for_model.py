# -*- coding: utf-8 -*-
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult


# getting x and y from results returned by handlandmarker.task model
# also handeness
def get_row_for_model(detection_result: HandLandmarkerResult):
    hand_landmarks = detection_result.hand_landmarks[0]
    handedness = detection_result.handedness[0]
    ret_dict = dict()
    row_lst = []

    for landmark in hand_landmarks:
        row_lst.append(landmark.x)
        row_lst.append(landmark.y)

    ret_dict["row"] = row_lst
    ret_dict["handedness"] = handedness

    return ret_dict
