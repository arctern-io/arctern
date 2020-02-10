with open('/tmp/points.json', 'w') as file:
    for i in range(10):
        file.write('{"x": %f, "y": %f}\n' % (i + 0.1, i + 0.1))
