# -*- coding: utf-8 -*-
import os.path

import numpy as np
import cv2

# function for converting video from mp4 to avi
def convert_video(path: str, path_to_save_dir: str):
    filename = path

    path_lst = path.split("\\")
    # file_output = ""
    #
    # for it in path_lst[:-1]:
    #     file_output = os.path.join(file_output, it)

    file_output_name = path_lst[-1].split(".")[0] + ".avi"
    file_output = os.path.join(path_to_save_dir, file_output_name)

    # load the file
    cap = cv2.VideoCapture(filename)

    # check if file is opened
    if cap.isOpened() == False:
        print("Error opening the file")

    # flip the image left-right
    # def flip_image(image):
    #     return cv2.flip(image, 1)

    # create video writer object using the sitting of the input video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    # fourcc = cv2.cv.CV_FOURCC(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_output, fourcc, float(fps), (frame_width, frame_height))

    # frames = 0

    # read until end of video
    while cap.isOpened():
        # compute first 100 frames
        # frames += 1
        # if frames == 100:
        #     break

        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            # perform the flip
            # flip_frame = flip_image(frame)
            # write the flipped frame
            out.write(frame)
            # display the frame
            # cv2.imshow('frame', flip_frame)
            # # press `q` to exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        # if no frame found
        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # close the video writer
    out.release()
    print("The video was successfully saved")
