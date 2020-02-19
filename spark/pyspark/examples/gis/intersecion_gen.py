with open('/tmp/intersection.json', 'w') as file:
    file.write('{"left": "POINT(0 0)", "right": "LINESTRING ( 2 0, 0 2  )"}\n')
    file.write('{"left": "POINT(0 0)", "right": "LINESTRING ( 0 0, 0 2  )"}\n')
