class ASRKeyboard:
    def __init__(self, bg_capture_keys, bg_reset_keys, exit_keys, gather_mode_keys):
        self.__bg_capture_keys = bg_capture_keys
        self.__bg_reset_keys = bg_reset_keys
        self.__exit_keys = exit_keys
        self.__gather_mode_keys = gather_mode_keys

    def is_bg_capture_key(self, key: int) -> bool:
        return key in self.__bg_capture_keys

    def is_bg_reset_key(self, key: int) -> bool:
        return key in self.__bg_reset_keys

    def is_exit_key(self, key: int) -> bool:
        return key in self.__exit_keys

    def is_gather_mode_key(self, key: int) -> bool:
        return key in self.__gather_mode_keys
