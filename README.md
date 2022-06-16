# Analysis of atomic swaps using MJD (Merton Jump Diffusion)
Install the dependencies:
```
sudo apt-get install texlive-latex-base texlive-latex-extra -y
pip3 install scipy matplotlib 
```

Use the code in the following way:

```
rm -rf swap_output && rm -rf limits.xml && python3 run.py 30000 steem_converted
```
steem_converted format: [UNIX TIMESTAMP][PRICE][INDEX 1h][INDEX 1s]
use convert_prices.py for this purpose
