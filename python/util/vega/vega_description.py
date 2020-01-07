class VegaDescription:
    def __init__(self, desc=""):
        self._description = desc

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    def __str__(self):
        return self.description