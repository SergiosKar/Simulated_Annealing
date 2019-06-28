from Sim_anneal import SimAnneal
import csv
import random
import time

# From new branch

'''
#use this to open cities and cities2 datasets with 45 and 48 cities
coords = []
with open('cities.txt','r') as f:
    i = 0
    for line in f.readlines():
        line = [float(x.replace('\n','')) for x in line.split(' ')]
        coords.append([])
        for j in range(1,3):
            coords[i].append(line[j])
        i += 1


coords=coords[0:30]
'''

#use this to read world cities datasets with 7322 cities
with open('world_cities.csv', 'rt',encoding="utf8") as f:
    reader = csv.reader(f)
    coords = [[float(row[2]), float(row[3]),row[0],row[5]] for row in reader]
#select N random entries
random.seed(9001)
coords=random.sample(coords,50)
#coords=coords[0:1000]


if __name__ == '__main__':
    start = time.time()
    sa = SimAnneal(coords, stop_iterat = 10000)

    sa.Anneal()
    end=time.time()
    print("Exexution time = ",end-start,"s")
    sa.printTour(flag=1)#flag =0 gia to cities and flag =1 gia to worldcities
    sa.visualizePath()
    sa.plotLearning()

