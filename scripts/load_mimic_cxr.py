import csv


with open('../../mimic-cxr/train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'train/p10524315/s01/view1_frontal.jpg':
            print row