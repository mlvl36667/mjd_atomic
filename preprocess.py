#####################################################################
#### Developed by Bence Ladoczki <ladoczki@tmit.bme.hu> 2022 Maj ####
####               All rights reserved                           ####
#####################################################################
import sys
import numpy as np
import json
import datetime
import time
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad


rates = []
with open('bifi_prices') as f:
 for line in f:
  x, y = line.split(",")
  rates.append([int(x),float(y)])

prev=-1
delta_hist={}
for x,y in rates:
 #print(x,prev)
 if prev!=-1:
    if x-prev>=10000:
        print('big skip at',x,'prev:',prev,'diff:',(x-prev)/1000,'sec')
    else:
        try:
            delta_hist[x-prev] += 1
        except KeyError:
            delta_hist[x-prev] = 1
 prev=x
avg_delta=(rates[-1][0]-rates[0][0])/len(rates)
print('avg delta=',avg_delta)

delta=1000

#print(delta)

plt.clf()
plt.xlabel(r'delta time')
plt.ylabel('number of')
plt.grid(axis='y', color='0.95')
plt.bar(delta_hist.keys(), delta_hist.values(), 1.0, color='g')
plt.yscale("log")
plt.show()

price_output = open("bifi_price_list", "w")
time=rates[0][0]//delta
time0=time
for x,y in rates:
    if x>time*delta:
        dx= x// delta
        #print('dx=',dx,'time',time*delta)
        for i in range(dx-time):
            price_output.write(str(y)+"\n")#+' at '+str(x)
            time+=1
print('time:',time-time0)
price_output.close()
