import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult


def get_row_for_model(det_result: HandLandmarkerResult):
    hand_landmarks = det_result.hand_landmarks[0]
    handedness = det_result.handedness[0]
    ret_dict = dict()
    row_lst = []

    for landmark in hand_landmarks:
        row_lst.append(landmark.x)
        row_lst.append(landmark.y)

    ret_dict["row"] = row_lst
    ret_dict["handedness"] = handedness

    return ret_dict