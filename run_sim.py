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

import argparse

aparser = argparse.ArgumentParser()
aparser.add_argument("-fig", help="The output is a figure", action='store_true')
aparser.add_argument("-short_time", type=int, help="Parameter t_0 in [sec]", default=1)
aparser.add_argument("-price_avg", type=int, help="Use averaging of prices", default=1)
aparser.add_argument("-parse_till", type=int, help="Parse the rate time series till", default=-1)
aparser.add_argument("-tau1", type=float, help="Parameter tau_1 in [hour]", default=1)
aparser.add_argument("-tau2", type=float, help="Parameter tau_2 in [hour]", default=1)
aparser.add_argument("-alpha1", type=float, help="Parameter alpha_1", default=0.1)
aparser.add_argument("-alpha2", type=float, help="Parameter alpha_2", default=0.1)
aparser.add_argument("-betaA", type=float, help="Parameter beta_A", default=0.1)
aparser.add_argument("-betaB", type=float, help="Parameter beta_B", default=0.1)
aparser.add_argument("-ra", type=float, help="Parameter r_a", default=0.01)
aparser.add_argument("-rb", type=float, help="Parameter r_b", default=0.01)
aparser.add_argument("-price_file", type=str, help="The file with exchange rates. Each line corresponds to a second", default='bifi_price_list')
aparser.add_argument("-xml_outfile", type=str, help="The xml file where the results are stored", default='limits.xml')
aparser.add_argument("-epsilon2", type=int, help="Paramter epsilon_2", default=1)
prediction = aparser.add_mutually_exclusive_group()
prediction.add_argument("-bs", help="The price is estimated by Black-Scholes formula (default)", action='store_true')
prediction.add_argument("-mjd", help="The price is estimated by Merton Jump Diffusion Model ", action='store_true')
aparser.add_argument("-jump_criteria", type=float, help="Parameter jump_criteria of MJD", default=0.1)
aparser.add_argument("-fig_rate", help="Print data for a chart of the rate prediction algorithm", action='store_true')

args = aparser.parse_args()

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

short_time = args.short_time # in s
tau1 = args.tau1*60*60 # in s
tau2 = args.tau2*60*60 # in s
alpha1 = args.alpha1
alpha2 = args.alpha2
ra = args.ra
rb = args.rb
epsilon2 = args.epsilon2
price_average=args.price_avg

def log_string(file_descriptor,string):
    file_descriptor.write(string+"\n")

def get_coin_price(t):
    global rates
    if(t<len(rates)):
        return rates[t]
    print("Price not found for "+str(t)+", returning the last..."+str(len(rates)))
    return rates[-1]

def get_coin_price_range(t):
    global price_average, rates
    if(t+price_average<=len(rates)):
        sum_rate=0
        i=0
        for rate in rates[t:t+price_average]:
            sum_rate+=rate
            i+=1
        return sum_rate/i
    print("Price not found for "+str(t)+", returning the last..."+str(len(rates)))
    return rates[-1]

def estimate_coin_price(at, to):
    """ Returns the estimated coin price of the time to, if the estimation is done at time
    :param at: The time point when the estimation is done
    :param to: The time point of to estimate the time
    :return: The estimated price"""
    global args, sigma_hat, mu_hat, sigma_jhat, mu_jhat, lambda_hat
    p_at=get_coin_price_range(at)
    tau=(to-at)/3600
    if args.mjd:
        # estimated by Merton Jump Diffusion Model (MJD) formula
        return p_at * math.exp( (mu_hat - sigma_hat**2 / 2 )*tau + lambda_hat*mu_jhat*tau )
        mjd_mu = mu_jhat
        mjd_sigma = sigma_jhat
        mjd_lambda = lambda_hat
        #R_t e^{\mu (t'-t)} e^{\lambda \tau ( e^{\mu_j + \frac{\sigma^2_j}{2}} - 1 ) }
        return p_at * math.exp(mu_hat*(to-at)/3600) *math.exp( mjd_lambda  * ((to-at)/3600) * (math.exp( mjd_mu + mjd_sigma**2/2) - 1) )
    #if args.bs:
    # estimated by Black-Scholes formula
    #print(p_at, mu_hat, sigma_hat, to, at,(mu_hat-sigma_hat**2/2)*(to-at)/3600)
    # R_t e^{\mu(t'-t)}
    return p_at * math.exp((mu_hat)*tau )

# global parameters:
length_of_sim=0
sigma_hat=0
mu_hat=0
sigma_jhat=0
lambda_hat=0
def compute_parameters(jump_criteria = 0.05, till=-1):
    global args, rates, length_of_sim, sigma_hat, mu_hat, sigma_jhat, mu_jhat
    log_rates=[math.log(x) for x in rates]
    if till==-1:
        diff = np.diff(log_rates)
    else:
        diff = np.diff(log_rates[max(a,till-1000):till])

    diff_list = diff.tolist()


    if args.mjd:
        max_jump=np.max(diff)
        log_string(swap_output, "Maximum of jumps: "+str(max_jump))
        log_string(swap_output, "Minimum of jumps: "+str(np.min(diff)))
        jump_criteria=min(max_jump,jump_criteria)
        length_of_sim = len(rates) / (60*60)

        d_t = 1 /3600

        significant_jumps = [x for x in diff_list if abs(x) >= jump_criteria]
        bs_terms = [x for x in diff_list if abs(x) < jump_criteria ] # and abs(x) > 0.000001

        lambda_hat = len(significant_jumps) / length_of_sim
        log_string(swap_output, "lambda_hat: "+str(lambda_hat))
        log_string(swap_output, "Length of sim: "+str(length_of_sim)+" hours, which is "+str(len(rates))+" data points.")
        log_string(swap_output, "It is divided into Significant jumps and B-S terms as follows:")
        log_string(swap_output, "Significant jumps: "+str(significant_jumps)+" len: "+str(len(significant_jumps)))
        print('Number of jumps identified in MJD:',len(significant_jumps))
        log_string(swap_output, "B-S terms: "+str(bs_terms)+" len: "+str(len(bs_terms)))

        sigma_hat = np.std(bs_terms) / math.sqrt(d_t)
        mu_hat = (2*np.mean(bs_terms) + sigma_hat*sigma_hat*d_t) / ( 2 * d_t)

        log_string(swap_output, "var_sign: "+str(np.var(significant_jumps)))

        sigma_jhat = math.sqrt(np.var(significant_jumps) - sigma_hat*sigma_hat * d_t)

        mu_jhat = np.mean(significant_jumps) - ( mu_hat - sigma_hat*sigma_hat / 2) * d_t

        log_string(swap_output, "-------------------")
        log_string(swap_output, "mu_hat: "+str(mu_hat)+" (s)")
        log_string(swap_output, "sigma_hat: "+str(sigma_hat)+" (sqrt(s))")
        log_string(swap_output, "mu_jhat: "+str(mu_jhat)+" (s)")
        log_string(swap_output, "sigma_jhat: "+str(sigma_jhat)+" (s)")
        log_string(swap_output, "-------------------")
        print('sigma_hat:',sigma_hat,'mu_hat:',mu_hat,'sigma_jhat',sigma_jhat,'mu_jhat:',mu_jhat,'lambda_hat',lambda_hat)
        return
    # the default is bs model
    if args.bs or True:
        #######################################################
        ### Calculate Parameters For Black-Scholes         ####
        #######################################################

        mean_rate=np.mean(rates)
        std_rate=np.std(rates)

        d_t = 1 /3600
        #print('d_t=',d_t)

        variance_diff= np.std(diff)
        sigma_hat = math.sqrt(variance_diff / d_t)
        mu_hat =  np.mean(diff_list)/d_t +  variance_diff / 2 * d_t

        log_string(swap_output, "-------------------")
        log_string(swap_output, "sigma_hat_gbm: "+str(sigma_hat))
        log_string(swap_output, "mu_hat_gbm: "+str(mu_hat))
        print('sigma_hat:',sigma_hat,'mu_hat:',mu_hat)
        return
#def mjd_expectation(tau, pt, lmb,  mjd_mu, mjd_sigma, gbm_mu):
#    return pt*math.exp(gbm_mu*tau)*math.exp( lmb * tau* (math.exp( mjd_mu + mjd_sigma**2/2) - 1) )

#--------------------------------------------------------------------------------
def swap_coins_range(rates,t_0,swap_output, xml_output):
    global short_time, tau1, tau2, args

    pt_0= get_coin_price(t_0)
    t_1 = t_0 + short_time
    t_2 = t_1 + tau1
    t_3 = t_2 + tau2

    pt_3a_eq = (1+alpha1) * estimate_coin_price(t_3, t_0)
    print(t_3,'time:',estimate_coin_price(t_3, t_0),pt_0)
    log_string(xml_output, "<max5>"+str(pt_3a_eq)+"</max5>")

    pt_2b_eq = estimate_coin_price(t_2, t_0) / (1+alpha1)
    log_string(xml_output, "<min7>"+str(pt_2b_eq)+"</min7>")

    min_rate=pt_2b_eq #max(pt_2b_eq)
    max_rate=pt_3a_eq #min(pt_3a_eq)

    # printing an illustrative figure
    if args.fig_rate:
        log_string(xml_output, "<t0>"+str(t_0)+"</t0>")
        for t, rate in enumerate(rates[t_0:max(args.parse_till,t_3+tau2)]):
            if t % 10 == 0:
                log_string(xml_output, "<data><time>"+str(t)+"</time><rel_rate>"+str(rate)+"</rel_rate><est_rate>"+str(estimate_coin_price(t_0, t_0+t))+"</est_rate></data>")
        log_string(xml_output, "</datapoint></simulation>")
        sys.exit(0)

    log_string(xml_output, "<max_rate>"+str(max_rate)+"</max_rate>")
    log_string(xml_output, "<min_rate>"+str(min_rate)+"</min_rate>")
    log_string(xml_output, "<max_rel_rate>"+str(max_rate/pt_0)+"</max_rel_rate>")
    log_string(xml_output, "<min_rel_rate>"+str(min_rate/pt_0)+"</min_rel_rate>")
    log_string(xml_output, "<beta>"+str((max_rate-min_rate)/pt_0)+"</beta>")
    log_string(xml_output, "<opt_rel_rate>"+str(((max_rate+min_rate)/2-pt_0)/pt_0)+"</opt_rel_rate>")
    log_string(xml_output, "<rel_rate>"+str((max(max_rate-pt_0,pt_0-min_rate))/pt_0)+"</rel_rate>")
    if min_rate>max_rate:
     print('infeasible range:',min_rate,max_rate)
     return False
    else:
     swap_coins(rates,t_0,swap_output, min_rate+0.01, 0.01, xml_output)
     swap_coins(rates,t_0,swap_output, max_rate-0.01, 0.01, xml_output)
     return True

def swap_coins_range_old(rates,t_0,swap_output, xml_output):
    global short_time, tau1, tau2, alpha1, alpha2, ra, rb, epsilon2, swap_success

    tau1_h = tau1 / (60*60)
    tau2_h = tau2 / (60*60)

    pt_0 = get_coin_price(t_0)
    t_1 = t_0 + short_time
    pt_1 = get_coin_price_range(t_1)
    t_2 = t_1 + tau1
    pt_2 = get_coin_price_range(t_2)
    t_3 = t_2 + tau2
    pt_3 = get_coin_price_range(t_3)
    t_4 = t_3 + epsilon2
    pt_4 = get_coin_price_range(t_4)
    t_5 = t_3 + tau2
    pt_5 = get_coin_price_range(t_5)
    t_6 = t_4 + tau1
    pt_6 = get_coin_price_range(t_6)
    t_7 = t_3 + tau2 + tau2
    pt_7 = get_coin_price_range(t_7)

    ## t2
    #ut3acont = (1+alpha1)* pt_5 / math.exp(ra*tau2_h)
    #ut3astop = exchange_rate / math.exp(ra*(epsilon2+2*tau1_h))
    # ut3acont==ut3astop mennyi az exchange_rate
    #pt_5a_eq = ((1+alpha1)* pt_5 / math.exp(ra*tau2_h)) * math.exp(ra*(epsilon2+2*tau1_h))
    pt_3a_eq = ((1+alpha1)* pt_3 / math.exp(ra*tau2_h)) * math.exp(ra*(epsilon2+2*tau1_h))
    log_string(xml_output, "<max5>"+str(pt_3a_eq)+"</max5>")


    # ut3bcont = (1+alpha2) * exchange_rate / math.exp(rb*(epsilon2+tau1_h))
    ut3bstop = pt_7 / math.exp(ra*(epsilon2+2*tau1_h))
    # ut3bcont==ut3bstop mennyi az exchange_rate
    pt_7b_eq = (pt_7 / math.exp(ra*(epsilon2+2*tau1_h))) * math.exp(rb*(epsilon2+tau1_h)) / (1+alpha2)
    log_string(xml_output, "<min7>"+str(pt_7b_eq)+"</min7>")

    # utility = ut3acont
    # ut2acont =  utility / math.exp(ra*tau2_h)
    # ut2astop = exchange_rate / math.exp(ra*(tau2_h+epsilon2+2*tau1_h))
    # ut2acont==ut2astop mennyi az exchange_rate
    pt_2a_eq = (ut3acont / math.exp(ra*tau2_h)) * math.exp(ra*(tau2_h+epsilon2+2*tau1_h))
    log_string(xml_output, "<max2>"+str(pt_2a_eq)+"</max2>")

    ut3bcont = ut3bstop
    # ut3bcont = (1+alpha2) * exchange_rate / math.exp(rb*(epsilon2+tau1_h))
    # utility = max(ut3bcont,ut3bstop)
    # ut2bcont = utility / math.exp(rb*tau2_h)
    pt_2b_eq = math.exp(rb*(epsilon2+tau1_h)) * math.exp(rb*tau2_h) / (1+alpha2)
    # # ut2bstop = pt_2
    # # ut2bcont > ut2bstop
    # if(pt_2 > ut3bcont / math.exp(rb*tau2_h)):
    #  log_string(xml_output, "<min_pt2_b>no_such_minimum_exists (pt_2: "+str(pt_2)+", ut3bcont: "+str(ut3bcont / math.exp(rb*tau2_h))+")</min_pt2_b>")
    #  pt_2b_eq = pt_2a_eq+0.01
    # else:
    #  pt_2b_eq = ut3bstop * math.exp(rb*tau2_h)
    #  log_string(xml_output, "<min_pt2_b>"+str(pt_2b_eq)+"</min_pt2_b>")
    # ### for debugging:
    # #pt_2b_eq = ut3bstop * math.exp(rb*tau2_h)
    ## t1
    #ut2acont =  ut3acont / math.exp(ra*tau2_h)
    #ut2astop = exchange_rate / math.exp(ra*(tau2_h+epsilon2+2*tau1_h))
    pt_1a_eq = (ut3acont / math.exp(ra*tau2_h)) * math.exp(ra*(tau2_h+epsilon2+2*tau1_h))

    log_string(xml_output, "<max1>"+str(pt_2a_eq)+"</max1>")
    min_rate=max(pt_7b_eq,pt_2b_eq)
    max_rate=min(pt_3a_eq,pt_2a_eq,pt_1a_eq)
    log_string(xml_output, "<max_rate>"+str(max_rate)+"</max_rate>")
    log_string(xml_output, "<min_rate>"+str(min_rate)+"</min_rate>")
    log_string(xml_output, "<max_rel_rate>"+str(max_rate/pt_0)+"</max_rel_rate>")
    log_string(xml_output, "<min_rel_rate>"+str(min_rate/pt_0)+"</min_rel_rate>")

    if min_rate>max_rate:
     print('infeasible range:',min_rate,max_rate)
     return False
    else:
     swap_coins(rates,t_0,swap_output, min_rate+0.01, 0.01, xml_output)
     swap_coins(rates,t_0,swap_output, max_rate-0.01, 0.01, xml_output)
     return True

def swap_coins(rates,t_0,swap_output, exchange_rate, price_delta, xml_output):
    global short_time, tau1, tau2, alpha1, alpha2, ra, rb, epsilon2, swap_success

    swap_success = True
    tau1_h = tau1 / (60*60)
    tau2_h = tau2 / (60*60)

    pt_0 = get_coin_price(t_0)
    t_1 = t_0 + short_time
    pt_1 = get_coin_price_range(t_1)
    t_2 = t_1 + tau1
    pt_2 = get_coin_price_range(t_2)
    t_3 = t_2 + tau2
    pt_3 = get_coin_price_range(t_3)
    t_4 = t_3 + epsilon2
    pt_4 = get_coin_price_range(t_4)
    t_5 = t_3 + tau2
    pt_5 = get_coin_price_range(t_5)
    t_6 = t_4 + tau1
    pt_6 = get_coin_price_range(t_6)
    t_7 = t_3 + tau2 + tau2
    pt_7 = get_coin_price_range(t_7)
    #
    # log_string(xml_output, "pt0: "+str(pt_1)+" t0: "+str(t_0))
    # log_string(xml_output, "pt1: "+str(pt_1))
    # log_string(xml_output, "pt2: "+str(pt_2))
    # log_string(xml_output, "pt3: "+str(pt_3))
    # log_string(xml_output, "pt4: "+str(pt_4))
    # log_string(xml_output, "pt5: "+str(pt_5)+" t5: "+str(t_5))
    # log_string(xml_output, "pt6: "+str(pt_6))
    # log_string(xml_output, "pt7: "+str(pt_7))

    ## t2
    ut3acont = (1+alpha1)* pt_5 / math.exp(ra*tau2_h)
    ut3astop = exchange_rate / math.exp(ra*(epsilon2+2*tau1_h))

    ut3bcont = (1+alpha2) * exchange_rate / math.exp(rb*(epsilon2+tau1_h))
    ut3bstop = pt_7 / math.exp(ra*(epsilon2+2*tau1_h))

    if(ut3acont < ut3astop):
        #  log_string(swap_output, "The swap fails at t3 due to A")
        log_string(xml_output, "<reason>a_aborts_at_t3:"+str(ut3acont)+"less"+str(ut3astop)+" (exchange_rate: "+str(exchange_rate)+")</reason>")
        swap_success = False
    # else:
    #  log_string(swap_output, "A would cont. at t3")

    utility = max(ut3acont , ut3astop)
    ## t2
    ut2acont =  utility / math.exp(ra*tau2_h)
    ut2astop = exchange_rate / math.exp(ra*(tau2_h+epsilon2+2*tau1_h))

    utility = max(ut3bcont,ut3bstop)

    ut2bcont = utility / math.exp(rb*tau2_h)
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

    ut1acont = utility / math.exp(ra*tau1_h)
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

############################################
####### Entry point to the script ##########
############################################

rates = []
with open(args.price_file) as f:
 for count,line in enumerate(f):
  rates.append(float(line))
  if count==args.parse_till:
      break

xml_output = open(args.xml_outfile, "w") # used to be "a"
log_string(xml_output, '<?xml version="1.0" encoding="UTF-8"?>')
log_string(xml_output, '<simulation>')
for arg in vars(args):
    log_string(xml_output, '<'+arg+'>'+str(getattr(args, arg))+'</'+arg+'>')
if args.mjd:
    log_string(xml_output, '<predict_method>MJD</predict_method>')
else:
    log_string(xml_output, '<predict_method>BS</predict_method>')
swap_output = open("swap_output.txt", "w")
compute_parameters(args.jump_criteria,-1)
log_string(swap_output, "------------------------")
log_string(swap_output, "Launching the atomic swap simulator at "+str(datetime.datetime.now()))
log_string(swap_output, "------------------------")

filter=0
filter_step=60
time=-1
succnum=0
failnum=0
for rate in tqdm(rates[:-4*60*60]):
    time+=1
    if time % filter_step!=0:
        continue
    log_string(xml_output, "<datapoint>")
    log_string(xml_output, "<rate>"+str(rate)+"</rate>")
    succeed=swap_coins_range(rates,time,swap_output, xml_output)
    if succeed:
        succnum+=1
    else:
        failnum+=1
    log_string(xml_output, "</datapoint>")

print('suceed',succnum,'failed',failnum)
log_string(xml_output, "<successrate>"+str(succnum/(succnum+failnum))+"</successrate>")
log_string(xml_output, '</simulation>')
xml_output.close()
swap_output.close()
