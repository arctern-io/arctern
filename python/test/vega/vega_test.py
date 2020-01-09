from python.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d

import json

def test_circle2d():
    vega_circle2d = VegaCircle2d(1900, 1410, 3, "#2DEF4A", 0.5)
    vega_json = vega_circle2d.build()
    vega_dict = json.loads(vega_json)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["shape"]["value"] == "circle"
    assert vega_dict["marks"][0]["encode"]["enter"]["stroke"]["value"] == "#2DEF4A"
    assert vega_dict["marks"][0]["encode"]["enter"]["strokeWidth"]["value"] == 3
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 0.5
    print("circle2d test pass")

if __name__ == "__main__":
    test_circle2d()
