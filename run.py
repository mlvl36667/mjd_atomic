#####################################################################
#### Developed by Bence Ladoczki <ladoczki@tmit.bme.hu> 2022 Maj ####
####               All rights reserved                           ####
#####################################################################
import sys
import numpy as np
import json
import datetime
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

# This function calculates the MJD density
def pdf_mjd(x, pt, kmax, lmb, tau, mjd_sigma, gbm_sigma, gbm_mu):
 pdfvalue = 0
 for k in range(0,kmax+1):
  first_term = (lmb*tau)**k / math.factorial(k)
  second_term = math.exp(-lmb * tau )
  third_term = 1/( math.sqrt(2*math.pi)*( math.sqrt(k)*mjd_sigma + math.sqrt(tau) * gbm_sigma ) )

  normal_term = math.exp((-1/2)*( ( math.log(x/pt) - k*mjd_sigma - tau*(gbm_mu - (gbm_sigma**2)/2) ) / (math.sqrt(k)*mjd_sigma + math.sqrt(tau) * gbm_sigma) )**2)
  increment = first_term * second_term * third_term * normal_term/x

#  print("---------------------")
#  print(str(x))
#  print(str(first_term))
#  print(str(second_term))
#  print(str(third_term))
#  print(str(normal_term))
#  print(str(increment))
#  print("---------------------")

  pdfvalue = pdfvalue + increment
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
 simulation_input = open('simulation_input.json')
 input_variables = json.load(simulation_input)

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
  
 plt.title("PDF and CDF in MJD")
 plt.xlabel('x')
 plt.ylabel('density')
 plt.yticks(yy)

# plt.plot(xx, yy)
# plt.plot(xx3, yy3)
#   
# plt.show()

 integral = quad(ut2bcontMJD_integrand, 0, pt3eq_mjd(2, mjd_lambda, ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, gbm_mu), args=(taub, rb, 2, kmax,  mjd_lambda, mjd_sigma, gbm_sigma, gbm_mu, mjd_mu))[0]
# print(str(integral))
 xx2 = []
 yy1 = []
 yy2 = []
 yy4 = []
 yy5 = []
 yy6 = []
 for i in range(1,30):
  yy1.append(ut2bcontMJD(2.4, i/10, kmax, 0.05, taub, mjd_sigma, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, mjd_mu, alphaa))
  yy4.append(ut2bcontMJD(2.4, i/10, kmax, 0.4, taub, mjd_sigma, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, mjd_mu, alphaa))
  yy6.append(ut2bcontMJD(2.4, i/10, kmax, 0.00001, taub, 0.00001, gbm_sigma, gbm_mu,rb, epsilonb, ra, alphab, taua, 0.00001, alphaa))
  yy5.append(i/10)
  xx2.append(i/10)

 plt.clf()
 plt.xlabel(r'$P(t_2)$ ')
 plt.ylabel(r'$U^{cont}_{B}(t_2)$ ')
 plt.yticks(np.arange(0.5, max(yy5), 0.5))
 plt.grid(axis='y', color='0.95')
# plt.plot(xx, yy)
 plt.plot(xx2, yy5)
 plt.plot(xx2, yy4, label=r'$ \lambda$ = 0.4', color="red")
 plt.plot(xx2, yy1, label=r'$ \lambda$ = 0.05', color="green")
 plt.plot(xx2, yy6, label="Black-Scholes", color="black")
 plt.legend()
   
 plt.show()
 plt.savefig('ut2bcont.pgf')
 
 xx2 = []
 yy1 = []
 yy2 = []
 yy6 = []
 for i in range(15,30):
  yy1.append(success_rate_MJD(i/10, pt0, kmax, 0.4, taua, mjd_sigma, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  mjd_mu, alphaa))
  yy2.append(success_rate_MJD(i/10, pt0, kmax, 0.05, taua, mjd_sigma, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  mjd_mu, alphaa))
  yy6.append(success_rate_MJD(i/10, pt0, kmax, 0.000001, taua, 0.00001, gbm_sigma, gbm_mu, taub, epsilonb, ra, rb, alphab,  0.00001, alphaa))
  xx2.append(i/10)

 plt.clf()
 plt.xlabel(r'$P^{*}$')
 plt.ylabel('$SR(P^{*})$')
 plt.grid(axis='y', color='0.95')
 plt.yticks(np.arange(0.1, max(yy6)+0.1, 0.1))
# plt.plot(xx, yy)
 plt.plot(xx2, yy1, label=r'$ \lambda$ = 0.4', color="red")
 plt.plot(xx2, yy2, label=r'$ \lambda$ = 0.05', color="green")
 plt.plot(xx2, yy6, label="Black-Scholes", color="black")
 plt.legend()
   
 plt.show()

 plt.savefig('sr.pgf')


 log_string(simulation_output, "------------------------")
 log_string(simulation_output, "Finalizing simulation output at "+str(datetime.datetime.now()))
 log_string(simulation_output, "------------------------")
 
 simulation_output.close()

run_simulation()
