class VegaWidth:
    def __init__(self, width=0):
        self._width = width

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width