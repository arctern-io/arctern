class VegaData:
    def __init__(self):
        self._name = ""
        self._url = ""

    @property
    def name(self):
        return self._name

    @property
    def url(self):
        return self._url

    @name.setter
    def name(self, name):
        self._name = name

    @url.setter
    def url(self, url):
        self._url = url