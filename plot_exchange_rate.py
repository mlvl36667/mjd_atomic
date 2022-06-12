######################################################
# Plot crypto exchange rates                ##########
# Developed by Bence Ladoczki <ladoczki@tmit.bme.hu> #
######################################################
import json 
import matplotlib.pyplot as plt
import numpy as np


plt.title("BIFI-USDT ticker (Binance)")
plt.ylabel('price')
plt.xlabel('time (h)')


axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)

yy = []
xx = []

first_element_set = False
with open('/home/c/bifi_prices') as f:
 for line in f:
  x, y = line.split(",")
  if(not first_element_set):
   first_element = int(x)
   first_element_set = True

  xx.append((int(x)-first_element)/(60*60*1000))
  yy.append(float(y))


plt.plot(xx, yy)
  
plt.show()
f.close()
