import threading

from detect_hand_gesture import detect_hand_gesture
from speech_recognition import recognise_speech_command


def main():
    f1 = threading.Thread(target=detect_hand_gesture)
    f2 = threading.Thread(target=recognise_speech_command)

    f1.start()
    f2.start()


if __name__ == '__main__':
    main()
