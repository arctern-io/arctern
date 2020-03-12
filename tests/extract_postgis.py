import sys

# data = sys.argv[1]
# sql = sys.argv[2]
# function_name = sys.argv[3]

sql_template_1 = "select %s('%s'::geometry);\n"
sql_template_2 = "select %s('%s'::geometry, '%s'::geometry);\n"

arr = []
# with open(data, 'r') as f:
#     lines = f.readlines()[1:]
#     for line in lines:
#         values = line.strip().split('|')
#         if len(values) == 1:
#             arr.append(sql_template_1 % (function_name, values[0]))

#         if len(values) == 2:
#             arr.append(sql_template_2 % (function_name, values[0], values[1]))


# with open(sql, 'w') as f: 
#     for line in arr:
#         f.writelines(line)

with open('st_crosses.out', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        if line.strip().startswith('st_crosses'):
            continue
        if line.strip().startswith('-----'):
            continue
        if line.strip().startswith('('):
            continue

        arr.append(line.strip() + '\n')

with open('st_crosses.out.new', 'w') as f:
    for e in arr:
        f.writelines(e)
