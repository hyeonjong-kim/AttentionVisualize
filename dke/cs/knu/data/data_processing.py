import csv

line_counter = 0
header = []
customer_list = []

f = open('dga_label.csv')
while 1:
    data = f.readline()
    data = f.readline().replace("\n", "")
    if line_counter == 50 : break
    if line_counter == 0:
        header = data.split(",") # 맨 첫 줄은 header로 저장
    else:
        customer_list.append(data.split(","))
    line_counter += 1

f1 = open('./dga_sample.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f1)
writer = csv.DictWriter(f1, fieldnames=["domain","source","class"])

for i in customer_list:
    wr.writerow(i)