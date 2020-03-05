import shapely
from shapely import wkt

filename = './expected/results/st_envelope.out'
# filename = './data/envelope.json'
# filename = './arctern_results/run_test_st_envelope.json'

with open(filename, 'r') as f:
    lines = f.readlines()
    print(lines)
    print(len(lines))
    for line in lines:
        # print(wkt.loads(line))
        print(line)
