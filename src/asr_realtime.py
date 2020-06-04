import time

import cv2
import numpy as np
from tensorflow_core.python.keras.models import load_model

from asr import AwesomeSignRecognizer
from keyboard import ASRKeyboard
import copy

ESC_KEY = 27
SPACE_KEY = 32

# Общие настройки
prediction = ''
action = ''
score = 0
selected_gesture = ''

# params
cap_region_x_begin = 0.5
cap_region_y_end = 0.8
bg_sub_threshold = 50
learningRate = 0
blur_value = 41
threshold = 60

thresh = None
gesture_names = {
    0: 'H',
    1: 'E',
    2: 'L',
    3: 'O',
    4: 'W',
    5: 'R',
    6: 'D'
}

# model = load_model('C:\Projects\AwesomeSignRecognition\models\saved_model.hdf5')
model = load_model('C:\Projects\AwesomeSignRecognition\models\VGG.h5')

def config_keyboard() -> ASRKeyboard:
    return ASRKeyboard(
        bg_capture_keys=[ord('b'), ord('B')],
        bg_reset_keys=[ord('r'), ord('R')],
        exit_keys=[ESC_KEY],
        gather_mode_keys=[ord('s'), ord('S')],
        prediction_mode_keys=[SPACE_KEY]
    )


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score


def main():
    global prediction
    global score
    global thresh
    global action

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
            # Вырезаем зону показа руки
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
            # Преобразуем изображение в двуцветное
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Отрисовываем текст
            cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255))
            cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255))
            cv2.imshow('ori', thresh)

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
        elif asr_keyboard.is_prediction_mode_key(key):
            cv2.imshow('original', frame)
            target = np.stack((thresh,) * 3, axis=-1)
            target = cv2.resize(target, (224, 224))
            target = target.reshape(1, 224, 224, 3)
            prediction, score = predict_rgb_image_vgg(target)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
