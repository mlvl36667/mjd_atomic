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
from tqdm import tqdm # progress bar

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

short_time = 1 # in ms if t_0 is a UNIX timestamp
taua = 60*60 # in ms
taub = 60*60 # in ms
alphaa = 0.01
alphab = 0.01
ra = 0.01
rb = 0.01
epsilonb = 1

def log_string(file_descriptor,string):
    file_descriptor.write(string+"\n")

def get_coin_price(t,rates):
 if(t<len(rates)):
   return rates[t]
 print("Price not found for "+str(t)+", returning zero..."+str(len(rates)))
 sys.exit(1)
 return 0

def get_coin_price_range(t,rates):
 if(t+10<len(rates)):
   sum_rate=0
   i=0
   for rate in rates[t:t+10]:
     sum_rate+=rate
     i+=1
   return sum_rate/10
 print("Price not found for "+str(t)+", returning zero..."+str(len(rates)))
 return 0

def swap_coins(rates,t_0,swap_output, exchange_rate, price_delta, xml_output):
 global short_time, taua, taub, alphaa, alphab, ra, rb, epsilonb, swap_success

 taua_h = taua / (60*60)
 taub_h = taub / (60*60)

 pt_0 = get_coin_price(t_0,rates)
 t_1 = t_0 + short_time
 pt_1 = get_coin_price_range(t_1,rates)
 t_2 = t_1 + taua
 pt_2 = get_coin_price_range(t_2,rates)
 t_3 = t_2 + taub
 pt_3 = get_coin_price_range(t_3,rates)
 t_4 = t_3 + epsilonb
 pt_4 = get_coin_price_range(t_4,rates)
 t_5 = t_3 + taub
 pt_5 = get_coin_price_range(t_5,rates)
 t_6 = t_4 + taua
 pt_6 = get_coin_price_range(t_6,rates)
 t_7 = t_3 + taub + taub
 pt_7 = get_coin_price_range(t_7,rates)
# log_string(swap_output, "pt0: "+str(pt_1)+" t0: "+str(t_0))
#
# log_string(swap_output, "pt1: "+str(pt_1))
# log_string(swap_output, "pt2: "+str(pt_2))
# log_string(swap_output, "pt3: "+str(pt_3))
# log_string(swap_output, "pt4: "+str(pt_4))
# log_string(swap_output, "pt5: "+str(pt_5)+" t5: "+str(t_5))
# log_string(swap_output, "pt6: "+str(pt_6))

## t2
 ut3acont = (1+alphaa)* pt_5 / math.exp(ra*taub_h)
 ut3astop = exchange_rate / math.exp(ra*(epsilonb+2*taua_h))

 ut3bcont = (1+alphab) * exchange_rate / math.exp(rb*(epsilonb+taua_h))
 ut3bstop = pt_7 / math.exp(ra*(epsilonb+2*taua_h))

 if(ut3acont < ut3astop):
#  log_string(swap_output, "The swap fails at t3 due to A")
  log_string(xml_output, "<reason>a_aborts_at_t3:"+str(ut3acont)+"less"+str(ut3astop)+" (exchange_rate: "+str(exchange_rate)+")</reason>")
  swap_success = False
# else:
#  log_string(swap_output, "A would cont. at t3")

 if(ut3acont > ut3astop): # ez így biztos hogy jó?
  utility = ut3acont
 else:
  utility = ut3astop

## t2
 ut2acont =  utility / math.exp(ra*taub_h)
 ut2astop = exchange_rate / math.exp(ra*(taub_h+epsilonb+2*taua_h))

 if(ut3bcont > ut3bstop): # ez így biztos hogy jó?
  utility = ut3bcont
 else:
  utility = ut3bstop

 ut2bcont = utility / math.exp(rb*taub_h)
 ut2bstop = pt_2

 if(ut2bcont < ut2bstop):
#  log_string(swap_output, "The swap fails at t2 due to B")
  swap_success = False
  log_string(xml_output, "<reason>b_aborts_at_t2:"+str(ut2bcont)+"less"+str(ut2bstop)+" (exchange_rate: "+str(exchange_rate)+")</reason>")
# else:
#  log_string(swap_output, "B would cont. at t2")
## t1

 if(ut2acont > ut2astop): # ez így biztos hogy jó?
  utility = ut2acont
 else:
  utility = ut2astop

 ut1acont = utility / math.exp(ra*taua_h)
 ut1astop = exchange_rate

 if(ut1acont < ut1astop):
#  log_string(swap_output, "The swap fails at t1 due to A")
  swap_success = False
  log_string(xml_output, "<reason>a_aborts_at_t1:"+str(ut1acont)+"less"+str(ut1astop)+" (exchange_rate: "+str(exchange_rate)+")</reason>")
# else:
#  log_string(swap_output, "A would cont. at t1")

 if(swap_success):
#  log_string(swap_output, "The swap succeeds for price_delta: "+str(price_delta))
  return 1
 else:
#  log_string(swap_output, "The swap fails for price_delta: "+str(price_delta))
  return 0
#--------------------------------------------------------------------------------
def swap_coins_range(rates,t_0,swap_output, xml_output):
 global short_time, taua, taub, alphaa, alphab, ra, rb, epsilonb, swap_success

 taua_h = taua / (60*60)
 taub_h = taub / (60*60)

 pt_0 = get_coin_price(t_0,rates)
 t_1 = t_0 + short_time
 pt_1 = get_coin_price_range(t_1,rates)
 t_2 = t_1 + taua
 pt_2 = get_coin_price_range(t_2,rates)
 t_3 = t_2 + taub
 pt_3 = get_coin_price_range(t_3,rates)
 t_4 = t_3 + epsilonb
 pt_4 = get_coin_price_range(t_4,rates)
 t_5 = t_3 + taub
 pt_5 = get_coin_price_range(t_5,rates)
 t_6 = t_4 + taua
 pt_6 = get_coin_price_range(t_6,rates)
 t_7 = t_3 + taub + taub
 pt_7 = get_coin_price_range(t_7,rates)

## t2
 ut3acont = (1+alphaa)* pt_5 / math.exp(ra*taub_h)
 #ut3astop = exchange_rate / math.exp(ra*(epsilonb+2*taua_h))
 # ut3acont==ut3astop mennyi az exchange_rate
 pt_5a_eq = ((1+alphaa)* pt_5 / math.exp(ra*taub_h)) * math.exp(ra*(epsilonb+2*taua_h))
 log_string(xml_output, "<max5>"+str(pt_5a_eq)+"</max5>")


# ut3bcont = (1+alphab) * exchange_rate / math.exp(rb*(epsilonb+taua_h))
 ut3bstop = pt_7 / math.exp(ra*(epsilonb+2*taua_h))
# ut3bcont==ut3bstop mennyi az exchange_rate
 pt_7b_eq = (pt_7 / math.exp(ra*(epsilonb+2*taua_h))) * math.exp(rb*(epsilonb+taua_h)) / (1+alphab)
 log_string(xml_output, "<min7>"+str(pt_7b_eq)+"</min7>")

 # utility = ut3acont
 # ut2acont =  utility / math.exp(ra*taub_h)
 # ut2astop = exchange_rate / math.exp(ra*(taub_h+epsilonb+2*taua_h))
 # ut2acont==ut2astop mennyi az exchange_rate
 pt_2a_eq = (ut3acont / math.exp(ra*taub_h)) * math.exp(ra*(taub_h+epsilonb+2*taua_h))
 log_string(xml_output, "<max2>"+str(pt_2a_eq)+"</max2>")

 ut3bcont = ut3bstop
 # ut2bcont = ut3bcont / math.exp(rb*taub_h)
 # ut2bstop = pt_2
 # ut2bcont > ut2bstop
 if(pt_2 > ut3bcont / math.exp(rb*taub_h)):
  log_string(xml_output, "<min_pt2_b>no_such_minimum_exists (pt_2: "+str(pt_2)+", ut3bcont: "+str(ut3bcont / math.exp(rb*taub_h))+")</min_pt2_b>")
  pt_2b_eq = pt_7b_eq
 else:
  pt_2b_eq = ut3bstop * math.exp(rb*taub_h)
  log_string(xml_output, "<min_pt2_b>"+str(pt_2b_eq)+"</min_pt2_b>")
## t1

 # utility = ut2acont
 # ut1acont = utility / math.exp(ra*taua_h)
 # ut1astop = exchange_rate
 #pt_1a_eq = (ut3acont / math.exp(ra*taub_h)) * math.exp(ra*(taub_h+epsilonb+2*taua_h))
 #log_string(xml_output, "<max>"+str(pt_2a_eq)+"</max>")
 min_rate=max(pt_7b_eq,pt_2b_eq)
 max_rate=min(pt_5a_eq,pt_2a_eq)
 log_string(xml_output, "<max>"+str(max_rate)+"</max>")
 log_string(xml_output, "<min>"+str(min_rate)+"</min>")
 if min_rate>max_rate:
     print('infeasible range:',min_rate,max_rate)
 swap_coins(rates,t_0,swap_output, min_rate+0.01, 0.01, xml_output)
 swap_coins(rates,t_0,swap_output, max_rate-0.01, 0.01, xml_output)
############################################
####### Entry point to the script ##########
############################################

rates = []
with open('bifi_price_list') as f:
 for line in f:
  rates.append(float(line))

xml_output = open("limits.xml", "w") # used to be "a"
log_string(xml_output, '<?xml version="1.0" encoding="UTF-8"?>')
log_string(xml_output, '<simulation>')

swap_output = open("swap_output", "a")
log_string(swap_output, "------------------------")
log_string(swap_output, "Launching the atomic swap simulator at "+str(datetime.datetime.now()))
log_string(swap_output, "------------------------")

log_string(swap_output, "Now calculating jumps")

rates1 = []
for rate in rates:
 rates1.append(rate)

diff = np.diff(rates1) / rates1[:-1]

diff_list = diff.tolist()

log_string(swap_output, "Maximum of jumps: "+str(np.max(diff)))
log_string(swap_output, "Minimum of jumps: "+str(np.min(diff)))
length_of_sim = (len(rates) / (60*60))

jump_criteria = 0.01

significant_jumps = [x for x in diff_list if abs(x) > jump_criteria]
bs_terms = [x for x in diff_list if abs(x) < jump_criteria]

intensity = len(significant_jumps) / length_of_sim
log_string(swap_output, "Intensity: "+str(intensity))
log_string(swap_output, "Lenth of sim: "+str(length_of_sim))

log_string(swap_output, "Significant jumps: "+str(significant_jumps)+" len: "+str(len(significant_jumps)))
sigma_hat = np.std(bs_terms)/math.sqrt(length_of_sim)
log_string(swap_output, "sigma_hat: "+str(sigma_hat))

mu_hat = (2*np.mean(bs_terms) + sigma_hat*sigma_hat*length_of_sim) / (2 * length_of_sim)

sigma_jhat = math.sqrt(np.var(significant_jumps) - sigma_hat*sigma_hat * length_of_sim)

mu_jhat = np.mean(significant_jumps) - ( mu_hat - sigma_hat*sigma_hat / 2) * length_of_sim

log_string(swap_output, "mu_hat: "+str(mu_hat))
log_string(swap_output, "sigma_jhat: "+str(sigma_jhat))
log_string(swap_output, "mu_jhat: "+str(mu_jhat))

#######################################################
### Use estimated data from historical ticker data ####
#######################################################

# gbm_mu = mu_hat
# gbm_sigma = sigma_hat
# taua = 3
# taub = 4
# alphaa = 0.2
# alphab = 0.2
# ra = 0.01
# rb = 0.01
# pt0 = 2
# epsilona = 1
# epsilonb = 1
#
# mjd_mu = mu_jhat
# mjd_sigma = sigma_jhat
# mjd_lambda = intensity
#
# kmax = 20
#
# ##############################################
# # print SR for real world data from Binance ##
# ##############################################
# xx2 = []
# yy1 = []
# yy2 = []
# yy6 = []
# yy7 = []
# for i in range(15,30):
#  yy1.append(success_rate_MJD(i/10, pt0, kmax, intensity, taua, mjd_sigma, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  mjd_mu, alphaa))
#  yy6.append(success_rate_MJD(i/10, pt0, kmax, 0.000001, taua, 0.00001, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  0.00001, alphaa))
#  xx2.append(i/10)


#######################################################################


#price_deltas = [ -0.14, -0.13, -0.12, -0.11,  -0.1, -0.05, 0, 0.01, 0.011, 0.012]
#npprices = np.array(price_deltas) * pt0 + np.array(price_deltas) + pt0
#price_deltas_3 = npprices.tolist()
#simulated_success_rate = []

#iter_length = 0
#for price_delta in price_deltas:
# start_time = time.time()
#
# number_of_successes = 0
# number_of_trials = 0
# number_of_datapoints = 0
# max_number_of_datapoints = 15000
#
# for rate in rates:
#  if(number_of_datapoints < max_number_of_datapoints):
#   success_or_failure = swap_coins(rates,rate[0],swap_output, rate[1] + price_delta*rate[1], price_delta)
#   number_of_trials += 1
#   number_of_datapoints += 1
#   number_of_successes = number_of_successes + success_or_failure
#
# log_string(swap_output, "Delta: "+str(price_delta)+", percentage:  "+str(number_of_successes/number_of_trials))
# simulated_success_rate.append( number_of_successes/number_of_trials )
#
# print(str(iter_length)+"/"+str(len(price_deltas))+" [--- %s seconds ---" % (time.time() - start_time))
# iter_length += 1

current = 0
filter=0
filter_step=60
time=-1
for rate in tqdm(rates[:-4*60*60]):
 time+=1
 if time % filter_step!=0:
     continue

 log_string(xml_output, "<datapoint>")
 log_string(xml_output, "<rate>"+str(rate)+"</rate>")

 swap_coins_range(rates,time,swap_output, xml_output)

 #price_delta = 0
 #while( swap_coins(rates,time,swap_output, rate + price_delta*rate, price_delta, xml_output) == 1 ):
 # price_delta = price_delta - 0.01

 # if(price_delta != 0):
 #  price_delta = price_delta + 0.01
 # log_string(xml_output, "<lower><pricedelta>"+str(price_delta)+"</pricedelta><pstar>"+str(rate + price_delta*rate)+"</pstar></lower>")
 #
 # price_delta = 0
 # while( swap_coins(rates,time,swap_output, rate + price_delta*rate, price_delta, xml_output) == 1 ):
 #  price_delta = price_delta + 0.01
 #
 # if(price_delta != 0):
 #  price_delta = price_delta - 0.01
 # log_string(xml_output, "<upper><pricedelta>"+str(price_delta)+"</pricedelta><pstar>"+str(rate + price_delta*rate)+"</pstar></upper>")

 # t_0 = time
 # short_time = 1 # in ms if t_0 is a UNIX timestamp
 # taua = 1*60*60 # in ms
 # taub = 1*60*60 # in ms
 # alphaa = 0.2
 # alphab = 0.2
 # ra = 0.01
 # rb = 0.01
 # epsilonb = 1
 #
 # taua_h = taua / (60*60)
 # taub_h = taub / (60*60)
 #
 # pt_0 = get_coin_price(t_0,rates)
 # t_1 = t_0 + short_time
 # pt_1 = get_coin_price_range(t_1,rates)
 # t_2 = t_1 + taua
 # pt_2 = get_coin_price_range(t_2,rates)
 # t_3 = t_2 + taub
 # pt_3 = get_coin_price_range(t_3,rates)
 # t_4 = t_3 + epsilonb
 # pt_4 = get_coin_price_range(t_4,rates)
 # t_5 = t_3 + taub
 # pt_5 = get_coin_price_range(t_5,rates)
 # t_6 = t_4 + taua
 # pt_6 = get_coin_price_range(t_6,rates)
 #
 # log_string(xml_output, "<prices><pt0>"+str(pt_0)+"</pt0><pt1>"+str(pt_1)+"</pt1><pt2>"+str(pt_2)+"</pt2><pt3>"+str(pt_3)+"</pt3><pt4>"+str(pt_4)+"</pt4><pt5>"+str(pt_5)+"</pt5></prices>")
 #
 log_string(xml_output, "</datapoint>")
 #print('simulate:'+str(current)+"/"+str(len(rates)//filter_step))
 current = current + 1

log_string(xml_output, '</simulation>')
xml_output.close()
sys.exit(0)

plt.clf()
plt.xlabel(r'$P^{*}$')
plt.ylabel('$SR(P^{*})$')
plt.grid(axis='y', color='0.95')
plt.yticks(np.arange(0.1, max(yy1)+0.1, 0.1))
# plt.plot(xx, yy)
#plt.plot(xx2, yy6, label="Black-Scholes", color="black")
plt.legend(title=r'Bifi-USDT Ticker (Binance) - SR($P_{*}$)')
#plt.plot(xx2, yy1, label=r'SR_{est} ', color="green")
plt.plot(price_deltas_3, simulated_success_rate, label=r'SR_{sim}$ ', color="blue")
plt.legend()


plt.savefig('real_world_sr.pgf')
plt.savefig("real_world_sr.pdf", bbox_inches='tight')


swap_output.close()

sys.exit(0)


############################################################
##### Simulate formulas from A Game-Theoretic Analysis of ##
##### Cross-Chain Atomic Swaps with HTLCs               ####

simulation_input = open('simulation_input.json')
input_variables = json.load(simulation_input)
simulation_input.close()

gbm_mu = input_variables['gbm_mu']
gbm_sigma = input_variables['gbm_sigma']
taua = input_variables['taua']
taub = input_variables['taub']
alphaa = input_variables['alphaa']
alphab = input_variables['alphab']
ra = input_variables['ra']
rb = input_variables['rb']
pt0 = input_variables['pt0']
epsilona = input_variables['epsilona']
epsilonb = input_variables['epsilonb']

mjd_mu = input_variables['mjd_mu']
mjd_sigma = input_variables['mjd_sigma']
mjd_lambda = input_variables['mjd_lambda']

kmax = input_variables['kmax']

run_simulation(gbm_mu, gbm_sigma, taua, taub, alphaa, alphab, ra, rb, pt0, epsilona, epsilonb, mjd_mu, mjd_sigma, mjd_lambda, kmax)
#############################################################
