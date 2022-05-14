#####################################################################
#### Developed by Bence Ladoczki <ladoczki@tmit.bme.hu> 2022 Maj ####
####               All rights reserved                           ####
#####################################################################
import sys
import json
import datetime
import math

def log_string(file_descriptor,string):
 file_descriptor.write(string+"\n")

def pt3eq_mjd(pstar, lmb,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, mu):
 first_term = pstar / math.exp(ra*(epsilonb + 2*taua))
 second_term = math.exp(ra*taub)/(1+alphaa)
 third_term = 1/( math.exp(mu*taub) * math.exp(lmb*taub*(math.exp(mjd_mu + mjd_sigma**2/2)-1))  )
 return first_term * second_term * third_term

def pdf_mjd(x, pt, kmax,lmb, tau , mjd_sigma, gbm_sigma, gbm_mu):
 pdfvalue = 0
 for k in range(0,kmax+1):
  normal_term = math.exp((-1/2)*( ( math.log(x/pt) - k*mjd_sigma - tau*(gbm_mu - gbm_sigma**2/2) ) / (math.sqrt(k)*mjd_sigma + math.sqrt(tau) * gbm_sigma) )**2)
  pdfvalue = pdfvalue + (lmb*tau)**k / math.factorial(k) * math.exp(-lmb * tau ) * 1/(math.sqrt(2*math.pi)*(math.sqrt(k)*mjd_sigma + math.sqrt(tau) * gbm_sigma))*normal_term/x
 return pdfvalue

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

 log_string(simulation_output, "------------------------")
 log_string(simulation_output, "Calculating MJD ")
 log_string(simulation_output, "------------------------")
 
 pstar = 1.6
 eq_price_at_t3 = pt3eq_mjd(pstar, mjd_lambda,ra, epsilonb, taua, taub, mjd_mu, mjd_sigma, alphaa, gbm_mu)

 log_string(simulation_output, "eq_price_at_t3: "+str(eq_price_at_t3))


 log_string(simulation_output, "------------------------")
 log_string(simulation_output, "Finalizing simulation output at "+str(datetime.datetime.now()))
 log_string(simulation_output, "------------------------")
 
 simulation_output.close()

run_simulation()
