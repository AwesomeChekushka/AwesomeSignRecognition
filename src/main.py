import cv2 as cv
import numpy as np
from keyboard import ASRKeyboard

ESC_KEY = 27

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width


def config_keyboard() -> ASRKeyboard:
    return ASRKeyboard(
        bg_capture_keys=[ord('b'), ord('B')],
        bg_reset_keys=[ord('r'), ord('R')],
        exit_keys=[ESC_KEY]
    )


def main():
    camera = cv.VideoCapture(0)
    asr_keyboard = config_keyboard()

    while camera.isOpened():
        # Capture frame-by-frame
        ret, frame = camera.read()

        frame = cv.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv.flip(frame, 1)  # flip the frame horizontally
        cv.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        cv.imshow('original', frame)

        key = cv.waitKey(10)
        print(key)
        if asr_keyboard.is_exit_key(key):
            break
        elif asr_keyboard.is_bg_capture_key(key):
            break
        elif asr_keyboard.is_bg_reset_key(key):
            break

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
