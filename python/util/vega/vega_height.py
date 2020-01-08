class VegaHeight:
    def __init__(self, height=0):
        self._height = height

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height