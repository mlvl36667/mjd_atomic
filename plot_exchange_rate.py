######################################################
# Plot crypto exchange rates                ##########
# Developed by Bence Ladoczki <ladoczki@tmit.bme.hu> #
######################################################
import json 
import matplotlib.pyplot as plt
import matplotlib

import numpy as np



matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


plt.clf()

plt.ylabel(r'price')
plt.xlabel(r'time (h)')



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
  
plt.savefig('bifiusdt.pgf')

plt.show()
f.close()
