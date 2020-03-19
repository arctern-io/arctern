import sys

result = sys.argv[1]
data = sys.argv[2]

with open(result, 'r') as f:
    lines = f.readlines()
    nums = [int(x.split(' ')[0]) for x in lines]

with open(data, 'r') as f:
    for (num, value) in enumerate(f, 1):
        if num - 1 in nums:
            print(value.strip())
