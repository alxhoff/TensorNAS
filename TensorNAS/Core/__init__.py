from enum import Enum


class EnumWithNone(str, Enum):
    def value(self):
        ret = self._value_
        if ret == "None":
            return None
        else:
            return ret
