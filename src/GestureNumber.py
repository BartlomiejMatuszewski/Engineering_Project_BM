# -*- coding: utf-8 -*-
import enum


#  defining gestures and number
class GestureNumber(enum.Enum):
    NEUTRAL = 0
    POINT_UP = 1
    POINT_DOWN = 2
    POINT_LEFT = 3
    POINT_RIGHT = 4
    ATTACK = 5
    BLOCK = 6


# utility function for getting number from gesture name, used in data preparation
# i needed category number based on name in video name (<person name>_<gesture name>.avi)
def gesture_category(filename: str):
    splitted_name = filename.split("_")
    name_avi = ""

    name_avi = splitted_name[1]

    for it in splitted_name[2:]:
        name_avi += "_"
        name_avi += it

    splitted_name = name_avi.split(".")
    name: str = splitted_name[0]

    gestures = ["NEUTRAL", "POINT_UP", "POINT_DOWN", "POINT_LEFT", "POINT_RIGHT", "ATTACK", "BLOCK"]

    output = None

    for it in gestures:
        if it.lower() in name:
            output = GestureNumber[it.upper()].value
            break

    return output

# print(GestureNumber(gesture_category()))
