import time

import cv2
import numpy as np

from asr import AwesomeSignRecognizer
from keyboard import ASRKeyboard
import copy

ESC_KEY = 27

# Общие настройки
prediction = ''
action = ''
score = 0
selected_gesture = 'D_'

# params
cap_region_x_begin = 0.5
cap_region_y_end = 0.8
bg_sub_threshold = 50
learningRate = 0
blur_value = 41
threshold = 60


def config_keyboard() -> ASRKeyboard:
    return ASRKeyboard(
        bg_capture_keys=[ord('b'), ord('B')],
        bg_reset_keys=[ord('r'), ord('R')],
        exit_keys=[ESC_KEY],
        gather_mode_keys=[ord('s'), ord('S')]
    )


def main():
    camera = cv2.VideoCapture(0)
    asr_keyboard = config_keyboard()
    asr = AwesomeSignRecognizer()
    cnt = 0

    while camera.isOpened():
        ret, frame = camera.read()

        # фильтр сглаживания
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        # Отразим кадр по горизонтали
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        cv2.imshow('original', frame)

        if asr.is_bg_captured:
            img = asr.remove_background(frame)
            # Вырезаем зону интереса
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
            # Преобразуем изображение в двуцветное
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Отрисовываем текст
            # cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 255, 255))
            # cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 255, 255))
            cv2.imshow('ori', thresh)

            if asr.gather_mode:
                if cnt < 8:
                    cnt += 1
                else:
                    asr.save_silhouette(selected_gesture, thresh)
                    print(f'Silhouette №{asr.img_counter} saved')
                    cnt = 0

        key = cv2.waitKey(10)

        if asr_keyboard.is_exit_key(key):
            break
        elif asr_keyboard.is_bg_reset_key(key):
            asr.reset_bg()
            print('Background reset')
        elif asr_keyboard.is_bg_capture_key(key):
            asr.capture_bg()
            print('Background capture')
        elif asr_keyboard.is_gather_mode_key(key):
            asr.gather_mode = not asr.gather_mode
            print('Gather mode: ' + 'on' if asr.gather_mode else 'off')

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
