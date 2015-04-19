import csv
in_file_path='101.csv'
out_file_path='101_c.csv'
csv_file=open(in_file_path, 'r')
cr=csv.reader(csv_file, delimiter=',')
a = {}
for r in cr:
    l = len(r) - 4
    zero = r[0]
    one = r[1]
    two = r[2]
    three = r[3]
    if not zero in a:
        a[zero] = {}
    if not one in a[zero]:
        a[zero][one] = {}
    if not two in a[zero][one]:
        a[zero][one][two] = {}
    if not three in a[zero][one][two]:
        a[zero][one][two][three] = []
    for i in range(l):
        a[zero][one][two][three].append(r[i + 4])

csv_file2=open(out_file_path, 'w')
cw = csv.writer(csv_file2, delimiter=",")
for i in a:
    for j in a[i]:
        for k in a[i][j]:
            for p in a[i][j][k]:
                cw.writerow([i] + [j] + [k] + [p] + a[i][j][k][p])
                
