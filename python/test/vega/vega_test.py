from python.util.vega.vega_node import *

import json

def test_node():
    width = Width(width=1900)
    height = Height(height=1410)
    description = Description(desc="circlesds_2d")
    data = Data(name="nyc_taxi", url="/data/0_5m_nyc_taxi.csv")
    domain1 = Scales.Scale.Domain(data="nyc_taxi", field="longitude_pickup")
    domain2 = Scales.Scale.Domain(data="nyc_taxi", field="latitude_pickup")
    scale1 = Scales.Scale(name="x", type="linear", domain=domain1)
    scale2 = Scales.Scale(name="y", type="linear", domain=domain2)
    scales = Scales([scale1, scale2])
    encode = Marks.Encode(shape=Marks.Encode.Value("circle"), stroke=Marks.Encode.Value("#23f31a"),
                          strokeWidth=Marks.Encode.Value(3), opacity=Marks.Encode.Value(0.5))
    marks = Marks(encode)
    root = Root(width=width, height=height, description=description,
                data=data, scales=scales, marks=marks)

    root_json = json.dumps(root.to_dict(), indent=2, sort_keys=True)
    print(root_json)
    dic = json.loads(root_json)
    print(dic)


if __name__ == "__main__":
    test_node()
