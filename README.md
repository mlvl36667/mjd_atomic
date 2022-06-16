# Analysis of atomic swaps using MJD (Merton Jump Diffusion)
Install the dependencies:
```
sudo apt-get install texlive-latex-base texlive-latex-extra -y
pip3 install scipy matplotlib 
```

Use the code in the following way:

```
rm -rf swap_output && rm -rf limits.xml && python3 run.py [NUMBER OF DATAPOINTS] [CONVERTED INPUT]
```
converted input format: [UNIX TIMESTAMP],[PRICE],[INDEX 1h],[INDEX 1s]

use "convert_prices.py" for this purpose

then SR (Figure 6. in Stochastic analysis of the success rate in atomic swaps) will be printed to "real_world_sr.pdf"
