import pandas as pd

data = pd.read_csv('../data/dga_label.csv')

l = {}

for i in data.domain:
    if l.get(i) == None :
        l[i] = 1
    else:
        l[i] = l[i] + 1

for key in l:
    if l[key] > 1:
        print(key)