import time
import cv2
import numpy as np


class AwesomeSignRecognizer:
    # Параметры
    BG_SUB_THRESHOLD = 50
    LEARNING_RATE = 0

    img_counter = 0
    gather_mode = False

    __bg = None
    __is_bg_captured = False

    def reset_bg(self) -> None:
        time.sleep(1)
        self.__bg = None
        self.__is_bg_captured = False

    def capture_bg(self) -> None:
        self.__bg = cv2.createBackgroundSubtractorMOG2(0, self.BG_SUB_THRESHOLD)
        time.sleep(2)
        self.__is_bg_captured = True

    def save_silhouette(self, selected_gesture: str, thresh) -> None:
        img_path = f"../data/{selected_gesture}_{self.img_counter}.jpg"
        cv2.imwrite(img_path, thresh)
        self.img_counter += 1

    @property
    def is_bg_captured(self) -> bool:
        return self.__is_bg_captured

    def remove_background(self, frame):
        fg_mask = self.__bg.apply(frame, learningRate=self.LEARNING_RATE)
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fg_mask)
        return res
