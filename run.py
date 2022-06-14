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

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def log_string(file_descriptor,string):
 file_descriptor.write(string+"\n")

def success_rate_MJD_integrand(x,pstar, ra, epsilonb,mjd_mu, alphaa, taub, pt0, kmax, lmb, taua, mjd_sigma, gbm_sigma, gbm_mu):
 return pdf_mjd(x, pt0, kmax, lmb, taua, mjd_sigma, gbm_sigma, gbm_mu)*(1-cdf_mjd( pt3eq_mjd(pstar, lmb,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, gbm_mu), x, kmax, lmb, taub, mjd_sigma, gbm_sigma, gbm_mu))

def success_rate_MJD(pstar, pt0, kmax, lmb, taua, mjd_sigma, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  mjd_mu, alphaa ):

 found_low = False
 found_high = False
 return_value = []
 pt2_low = 0
 pt2_high = 0
 for i in range(10,400):
  funct_value = ut2bcontMJD(pstar, i/100, kmax, lmb, taub, mjd_sigma, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, mjd_mu, alphaa)

  if(not found_low and abs(funct_value - i/100) > 0.2):
   found_low = True
   pt2_low = i/100

  if(found_low and abs(funct_value - i/100) < 0.2):
   found_high = True
   pt2_high = i/100


 print("pt2_low:"+str(pt2_low))
 print("pt2_high:"+str(pt2_high))

 if(not found_low or not found_high):
  return 0 

 return quad(success_rate_MJD_integrand, pt2_low, pt2_high, args=(pstar,ra, epsilonb,mjd_mu, alphaa, taub, pt0, kmax, lmb, taua, mjd_sigma, gbm_sigma, gbm_mu))[0]
def mjd_expectation(tau, pt, lmb,  mjd_mu, mjd_sigma, gbm_mu):
 return pt*math.exp(gbm_mu*tau)*math.exp( lmb * tau* (math.exp( mjd_mu + mjd_sigma**2/2) - 1) )

def ut2bcontMJD_integrand(x,taub, rb, pt2, kmax, lmb, mjd_sigma, gbm_sigma, gbm_mu, mjd_mu):
 ut3bstop = mjd_expectation(2*taub,x, lmb,  mjd_mu, mjd_sigma, gbm_mu)/math.exp(rb*2*taub)
 return pdf_mjd(x, pt2, kmax, lmb, taub, mjd_sigma, gbm_sigma, gbm_mu)*ut3bstop

def ut2bcontMJD(pstar, pt2, kmax, lmb, taub, mjd_sigma, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, mjd_mu, alphaa):

 ut3bcont = pstar*(1+alphab)/(math.exp(rb*(epsilonb+ra) ))

 first_term = (1 - cdf_mjd(pt3eq_mjd(pstar, lmb,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, gbm_mu), pt2, kmax, lmb, taub, mjd_sigma, gbm_sigma, gbm_mu ) ) * ut3bcont

 second_term = quad(ut2bcontMJD_integrand, 0, pt3eq_mjd(pstar, lmb,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, gbm_mu), args=(taub, rb, pt2, kmax, lmb, mjd_sigma, gbm_sigma, gbm_mu, mjd_mu))[0]
# print("------------------")
# print(str(pt3eq_mjd(pstar, lmb,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, gbm_mu)))
# print(str(first_term))
# print(str(second_term))
# print("------------------")

 return (first_term + second_term) /  math.exp(rb*taub)
# return (first_term  ) /  math.exp(rb*taub)

def pt3eq_mjd(pstar, lmb,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, mu):
 first_term = pstar / math.exp(ra*(epsilonb + 2*taua))
 second_term = math.exp(ra*taub)/(1+alphaa)
 third_term = 1/( math.exp(mu*taub) * math.exp(lmb*taub*(math.exp(mjd_mu + mjd_sigma**2/2)-1))  )
 return first_term * second_term * third_term

# This function calculates the BS density
def pdf_bs(x, pt, tau, gbm_sigma, gbm_mu):
 pdfvalue = 1/( math.sqrt(2*math.pi)*( math.sqrt(tau) * gbm_sigma ) )
 pdfvalue = pdfvalue * math.exp((-1/2)*( ( math.log(x/pt) - tau*(gbm_mu - (gbm_sigma**2)/2) ) / (math.sqrt(tau) * gbm_sigma) )**2)
 return pdfvalue/x
# This function calculates the MJD density
def pdf_mjd(x, pt, kmax, lmb, tau, mjd_sigma, gbm_sigma, gbm_mu):
 pdfvalue = 0
 for k in range(0,kmax+1):
  first_term = (lmb*tau)**k / math.factorial(k)
  third_term = 1/( math.sqrt(2*math.pi)*( math.sqrt(k)*mjd_sigma + math.sqrt(tau) * gbm_sigma ) )
  normal_term = math.exp((-1/2)*( ( math.log(x/pt) - k*mjd_sigma - tau*(gbm_mu - (gbm_sigma**2)/2) ) / (math.sqrt(k)*mjd_sigma + math.sqrt(tau) * gbm_sigma) )**2)

  increment = first_term *  third_term * normal_term/x

#  print("---------------------")
#  print(str(x))
#  print(str(first_term))
#  print(str(third_term))
#  print(str(normal_term))
#  print(str(increment))
#  print("---------------------")

  pdfvalue = pdfvalue + increment
 pdfvalue = pdfvalue*math.exp(-lmb * tau )
# print("pdfvalue: "+str(pdfvalue))
 return pdfvalue

# This function calculates the MJD CDF
def cdf_mjd(x, pt, kmax, lmb, tau, mjd_sigma, gbm_sigma, gbm_mu):
 cdfvalue = 0
 for k in range(0,kmax+1):
  first_term = (lmb*tau)**k / math.factorial(k)
  second_term = math.exp(-lmb * tau )

  normal_term = math.erfc((-1/math.sqrt(2))*( math.log(x/pt) - k*mjd_sigma - tau*(gbm_mu - (gbm_sigma**2)/2) ) / (math.sqrt(k)*mjd_sigma + math.sqrt(tau) * gbm_sigma) )
  increment = first_term * second_term * normal_term/2

#  print("---------------------")
#  print(str(x))
#  print(str(first_term))
#  print(str(second_term))
#  print(str(third_term))
#  print(str(normal_term))
#  print(str(increment))
#  print("---------------------")

  cdfvalue = cdfvalue + increment
 return cdfvalue

def run_simulation():



 simulation_output = open("simulation_output", "a")
 
 log_string(simulation_output, "------------------------")
 log_string(simulation_output, "Launching the MJD simulator at "+str(datetime.datetime.now()))
 log_string(simulation_output, "------------------------")
 
 log_string(simulation_output, "GBM Parameters ")

 log_string(simulation_output, "gbm_mu: "+str(gbm_mu))
 log_string(simulation_output, "gbm_sigma: "+str(gbm_sigma))
 log_string(simulation_output, "taua: "+str(taua))
 log_string(simulation_output, "taub: "+str(taub))
 log_string(simulation_output, "alphaa: "+str(alphaa))
 log_string(simulation_output, "alphab: "+str(alphab))
 log_string(simulation_output, "ra: "+str(ra))
 log_string(simulation_output, "rb: "+str(rb))
 log_string(simulation_output, "pt0: "+str(pt0))
 log_string(simulation_output, "epsilona: "+str(epsilona))
 log_string(simulation_output, "epsilonb: "+str(epsilonb))

 log_string(simulation_output, "MJD Parameters ")

 log_string(simulation_output, "mjd_mu: "+str(mjd_mu))
 log_string(simulation_output, "mjd_sigma: "+str(mjd_sigma))
 log_string(simulation_output, "mjd_lambda: "+str(mjd_lambda))
 log_string(simulation_output, "kmax: "+str(kmax))

 log_string(simulation_output, "------------------------")
 log_string(simulation_output, "Calculating MJD ")
 log_string(simulation_output, "------------------------")
 
 pstar = 1.6
 eq_price_at_t3 = pt3eq_mjd(pstar, mjd_lambda,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, gbm_mu)

 log_string(simulation_output, "eq_price_at_t3: "+str(eq_price_at_t3))

 xx = []
 yy = []
 for i in range(10,50):
  yy.append(pdf_mjd(i/10, pt0, kmax, mjd_lambda, taub, mjd_sigma, gbm_sigma, gbm_mu))
  xx.append(i/10)
  
 xx3 = []
 yy3 = []
 for i in range(10,50):
  yy3.append(cdf_mjd(i/10, pt0, kmax, mjd_lambda, taub, mjd_sigma, gbm_sigma, gbm_mu))
  xx3.append(i/10)
  
#######################
# print the integrand
#######################
 xx2 = []
 yy1 = []
 yy2 = []
 yy3 = []
 yy4 = []
 yy5 = []
 yy6 = []
 for i in range(10,350):
  yy1.append(success_rate_MJD_integrand(i/100,2.4, ra, epsilonb,mjd_mu, alphaa, taub, pt0, kmax, 0.05, taua, mjd_sigma, gbm_sigma, gbm_mu))
  yy2.append(success_rate_MJD_integrand(i/100,2.4, ra, epsilonb,mjd_mu, alphaa, taub, pt0, kmax, 0.1, taua, mjd_sigma, gbm_sigma, gbm_mu))
  yy3.append(success_rate_MJD_integrand(i/100,2.4, ra, epsilonb,mjd_mu, alphaa, taub, pt0, kmax, 0.3, taua, mjd_sigma, gbm_sigma, gbm_mu))
  yy4.append(success_rate_MJD_integrand(i/100,2.4, ra, epsilonb,0     , alphaa, taub, pt0, kmax, 0, taua, 0, gbm_sigma, gbm_mu))
  yy6.append(success_rate_MJD_integrand(i/100,2.4, ra, epsilonb,0     , alphaa, taub, pt0, kmax, 0, taua, 0, 0.125, gbm_mu))
  yy5.append(pdf_bs(i/100, 2.4, taub, gbm_sigma, gbm_mu))
  xx2.append(i/100)

 plt.clf()
 plt.xlabel(r'$P_{t_2}$ ')
 plt.ylabel(r'$P(P_{t_2},P_{t_1},\tau_a) [1 - C(P_{t_3}(P^{*}),P_{t_2},\tau_b)] $')
 plt.yticks(np.arange(0.5, max(yy4)+1, 0.2))
 plt.grid(axis='y', color='0.95')
 plt.plot(xx2, yy4, label="Black-Scholes", color="black")
 plt.plot(xx2, yy1, label=r'$ \lambda$ = 0.05', color="green")
 plt.plot(xx2, yy2, label=r'$ \lambda$ = 0.1', color="red")
 plt.plot(xx2, yy3, label=r'$ \lambda$ = 0.3', color="purple")
 plt.plot(xx2, yy6, label=r'$ \sigma$ = 0.125', color="orange")
# plt.plot(xx2, yy5, label=r'Real BS', color="yellow")
 plt.legend(title=r'Integrand of SR($P_{*}$)')
   
 plt.savefig("integrand.pdf", bbox_inches='tight')
 plt.savefig('integrand.pgf')
# sys.exit(0)

#######################
# print ut2bcont
#######################
 xx2 = []
 yy1 = []
 yy2 = []
 yy4 = []
 yy5 = []
 yy6 = []
 yy7 = []
 for i in range(1,35):
  yy1.append(ut2bcontMJD(2, i/10, kmax, 0.05, taub, mjd_sigma, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, mjd_mu, alphaa))
  yy4.append(ut2bcontMJD(2, i/10, kmax, 0.1, taub, mjd_sigma, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, mjd_mu, alphaa))
  yy7.append(ut2bcontMJD(2, i/10, kmax, 0.3, taub, mjd_sigma, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, mjd_mu, alphaa))
  yy6.append(ut2bcontMJD(2, i/10, kmax, 0.00001, taub, 0.00001, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, 0.00001, alphaa))
  yy5.append(i/10)
  xx2.append(i/10)

 plt.clf()
 plt.xlabel(r'$P(t_2)$ ')
 plt.ylabel(r'Utility')
 plt.yticks(np.arange(0.5, max(yy5), 0.5))
 plt.grid(axis='y', color='0.95')
# plt.plot(xx, yy)
 plt.plot(xx2, yy5)
 plt.plot(xx2, yy1, label=r'$ \lambda$ = 0.05', color="green")
 plt.plot(xx2, yy4, label=r'$ \lambda$ = 0.1', color="red")
 plt.plot(xx2, yy7, label=r'$ \lambda$ = 0.3', color="purple")
 plt.plot(xx2, yy6, label="Black-Scholes", color="black")
 plt.legend(title=r'$U^{cont}_{B}(t_2)$')
   
 plt.savefig('ut2bcont.pgf')
 plt.savefig("ut2bcont.pdf", bbox_inches='tight')
 
#######################
# print SR
#######################
 xx2 = []
 yy1 = []
 yy2 = []
 yy6 = []
 yy7 = []
 for i in range(15,30):
  yy1.append(success_rate_MJD(i/10, pt0, kmax, 0.1, taua, mjd_sigma, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  mjd_mu, alphaa))
  yy2.append(success_rate_MJD(i/10, pt0, kmax, 0.05, taua, mjd_sigma, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  mjd_mu, alphaa))
  yy7.append(success_rate_MJD(i/10, pt0, kmax, 0.3, taua, mjd_sigma, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  mjd_mu, alphaa))
  yy6.append(success_rate_MJD(i/10, pt0, kmax, 0.000001, taua, 0.00001, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  0.00001, alphaa))
  xx2.append(i/10)

 plt.clf()
 plt.xlabel(r'$P^{*}$')
 plt.ylabel('$SR(P^{*})$')
 plt.grid(axis='y', color='0.95')
 plt.yticks(np.arange(0.1, max(yy6)+0.1, 0.1))
# plt.plot(xx, yy)
 plt.plot(xx2, yy6, label="Black-Scholes", color="black")
 plt.plot(xx2, yy2, label=r'$ \lambda$ = 0.05', color="green")
 plt.plot(xx2, yy1, label=r'$ \lambda$ = 0.1', color="red")
 plt.plot(xx2, yy7, label=r'$ \lambda$ = 0.3', color="purple")
 plt.legend()
   

 plt.savefig('sr.pgf')
 plt.savefig("sr.pdf", bbox_inches='tight')

#######################################################################

 log_string(simulation_output, "------------------------")
 log_string(simulation_output, "Finalizing simulation output at "+str(datetime.datetime.now()))
 log_string(simulation_output, "------------------------")
 
 simulation_output.close()

def get_coin_price(t,rates):
 for rate in rates:
  if(rate[0] == t):
   return rate[1]
  if(rate[0] > t):
   return rate[1]
 print("Price not found for "+str(t)+", returning zero...")
 sys.exit(1)
 return 0

def swap_coins(rates,t_0,swap_output, exchange_rate, price_delta, xml_output):

 swap_success = True


 short_time = 1000 # in ms if t_0 is a UNIX timestamp
 taua = 3*60*60*1000 # in ms
 taub = 4*60*60*1000 # in ms
 alphaa = 0.2
 alphab = 0.2
 ra = 0.01
 rb = 0.01
 epsilonb = 1

 taua_h = taua / (60*60*1000)
 taub_h = taub / (60*60*1000)

 pt_0 = get_coin_price(t_0,rates)
 t_1 = t_0 + short_time
 pt_1 = get_coin_price(t_1,rates)
 t_2 = t_1 + taua
 pt_2 = get_coin_price(t_2,rates)
 t_3 = t_2 + taub
 pt_3 = get_coin_price(t_3,rates)
 t_4 = t_3 + epsilonb
 pt_4 = get_coin_price(t_4,rates)
 t_5 = t_3 + taub 
 pt_5 = get_coin_price(t_5,rates)
 t_6 = t_4 + taua 
 pt_6 = get_coin_price(t_6,rates)

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
 ut3bstop = get_coin_price(t_3 + taub + taub,rates) / math.exp(ra*(epsilonb+2*taua_h))

 if(ut3acont < ut3astop):
#  log_string(swap_output, "The swap fails at t3 due to A")
  log_string(xml_output, "<reason>a_aborts_at_t3</reason>")
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
  log_string(xml_output, "<reason>b_aborts_at_t2</reason>")
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
  log_string(xml_output, "<reason>a_aborts_at_t1</reason>")
# else:
#  log_string(swap_output, "A would cont. at t1")

 if(swap_success):
#  log_string(swap_output, "The swap succeeds for price_delta: "+str(price_delta))
  return 1
 else:
#  log_string(swap_output, "The swap fails for price_delta: "+str(price_delta))
  return 0
#--------------------------------------------------------------------------------

############################################
####### Entry point to the script ##########
############################################

rates = []
with open('/home/c/bifi_prices') as f:
 for line in f:
  x, y = line.split(",")
  rates.append([int(x),float(y)])

xml_output = open("limits.xml", "a")
log_string(xml_output, ' <?xml version="1.0" encoding="UTF-8"?>')

swap_output = open("swap_output", "a")
log_string(swap_output, "------------------------")
log_string(swap_output, "Launching the atomic swap simulator at "+str(datetime.datetime.now()))
log_string(swap_output, "------------------------")

log_string(swap_output, "Now calculating jumps")

rates1 = []
for rate in rates:
 rates1.append(rate[1])
 
diff = np.diff(rates1) / rates1[:-1]

diff_list = diff.tolist()

log_string(swap_output, "Maximum of jumps: "+str(np.max(diff)))
log_string(swap_output, "Minimum of jumps: "+str(np.min(diff)))
length_of_sim = (rates[-1][0]-rates[0][0]) / (1000*60*60)

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
for rate in rates:

 log_string(xml_output, "<datapoint>")
 log_string(xml_output, "<rate>"+str(rate[1])+"</rate>")

 price_delta = 0
 while( swap_coins(rates,rate[0],swap_output, rate[1] + price_delta*rate[1], price_delta, xml_output) == 1 ):
  price_delta = price_delta - 0.01



 if(price_delta != 0):
  price_delta = price_delta + 0.002
 log_string(xml_output, "<lower><pricedelta>"+str(price_delta)+"</pricedelta><pstar>"+str(rate[1] + price_delta*rate[1])+"</pstar></lower>")

 price_delta = 0
 while( swap_coins(rates,rate[0],swap_output, rate[1] + price_delta*rate[1], price_delta, xml_output) == 1 ):
  price_delta = price_delta + 0.002

 if(price_delta != 0):
  price_delta = price_delta - 0.002
 log_string(xml_output, "<upper><pricedelta>"+str(price_delta)+"</pricedelta><pstar>"+str(rate[1] + price_delta*rate[1])+"</pstar></upper>")

 t_0 = rate[0]
 short_time = 1000 # in ms if t_0 is a UNIX timestamp
 taua = 3*60*60*1000 # in ms
 taub = 4*60*60*1000 # in ms
 alphaa = 0.2
 alphab = 0.2
 ra = 0.01
 rb = 0.01
 epsilonb = 1

 taua_h = taua / (60*60*1000)
 taub_h = taub / (60*60*1000)

 pt_0 = get_coin_price(t_0,rates)
 t_1 = t_0 + short_time
 pt_1 = get_coin_price(t_1,rates)
 t_2 = t_1 + taua
 pt_2 = get_coin_price(t_2,rates)
 t_3 = t_2 + taub
 pt_3 = get_coin_price(t_3,rates)
 t_4 = t_3 + epsilonb
 pt_4 = get_coin_price(t_4,rates)
 t_5 = t_3 + taub
 pt_5 = get_coin_price(t_5,rates)
 t_6 = t_4 + taua
 pt_6 = get_coin_price(t_6,rates)

 log_string(xml_output, "<prices><pt0>"+str(pt_0)+"</pt0><pt1>"+str(pt_1)+"</pt1><pt2>"+str(pt_2)+"</pt2><pt3>"+str(pt_3)+"</pt3><pt4>"+str(pt_4)+"</pt4><pt5>"+str(pt_5)+"</pt5></prices>")

 log_string(xml_output, "</datapoint>")
 print(str(current)+"/"+str(len(rates)))
 current = current + 1
 
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

xml_output.close()
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
