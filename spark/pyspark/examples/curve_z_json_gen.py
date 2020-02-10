with open('/tmp/z_curve.json', 'w') as file:
    # y = 150
    for i in range(100, 200):
        file.write('{"x": %d, "y": %d}\n' % (i, 150))

    # y = x - 50
    for i in range(100, 200):
        file.write('{"x": %d, "y": %d}\n' % (i, i - 50))

    # y = 50
    for i in range(100, 200):
        file.write('{"x": %d, "y": %d}\n' % (i, 50))
