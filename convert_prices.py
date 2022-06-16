import sys
#######################################################################

rates = []
with open(str(sys.argv[1])) as f:
 for line in f:
  x, y, = line.split(",")
  rates.append([int(x),float(y),-1,-1])

# Data Transformation

 counter1 = 0
 for rate1 in rates:
 
  counter2 = 0
  found_1_h_delta = False
 
  found_1_s_delta = False
  for rate2 in rates[counter1:counter1+5000]:
 
   if(rate2[0]-rate1[0] > 60*60*1000 and not found_1_h_delta):
    found_1_h_delta = True
    rate1[2] = counter1+counter2 - 1
   counter2+=1
 
  counter2 = 0
  found_1_s_delta = False
  for rate2 in rates[counter1:counter1+10]:
 
   if(rate2[0]-rate1[0] > 1000 and not found_1_s_delta):
    found_1_s_delta = True
    rate1[3] = counter1+counter2 - 1
   counter2+=1
  counter1+=1

for rate in rates:
 print(str(rate[0])+","+str(rate[1])+","+str(rate[2])+","+str(rate[3]))

sys.exit(0)
