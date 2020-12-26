# Roulette
Automated NMR pulse sequence design program

Python program that allows user to design de novo, or refine, NMR pulse sequences for a chemical/biological system of choice.
The following publications are relevant to the developement this project

Original program:

Lapin, J., Nevzorov, A.A. (2020) De novo NMR pulse sequence design using Monte-Carlo optimization techniques. Journal of Magnetic Resonance, 310, 106641

Updated program:

Lapin, J., Nevzorov, A.A. (2020) Computer-Generated Pulse Sequences for 1H-15N and 1Hα-13Cα Separated Local Field Experiments, Journal of Magnetic Resonance, Journal of Magnetic Resonance, 317, 106794

There are two versions of the program, 1 that is specificially written to split calculations up between two GPUs (roulette.py,
plotroulette.py), and the other which is written for CPU/Numpy library (roulettenp.py, plotnp.py). 
