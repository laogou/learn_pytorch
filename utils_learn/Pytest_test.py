import math


class Rectangle:
    def __init__(self, length, width, size=(40, 20)):
        self.length = length
        self.width = width
        self._size = size

    def area(self):
        area = self.length * self.width
        return area

    def perimeter(self):
        perimeter = (self.length + self.width) * 2
        return perimeter

    def diff(self):
        diff = math.fabs(self.length - self.width)
        return diff

    def resize(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("illegal size")
        self._size = (width, height)

    def get_length(self):
        return self.length

    def get_width(self):
        return self.width