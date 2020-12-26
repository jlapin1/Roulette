# Roulette
Automated NMR pulse sequence design program

Python program that allows user to design de novo, or refine, NMR pulse sequences for a chemical/biological system of choice. The programs works
by optimizing a simulated NMR spectrum using a Monte Carlo/Simulated Annealing protocol to search through the pulse sequence parameters.

The following publications are relevant to the developement this project

Original program:

Lapin, J., Nevzorov, A.A. (2020) De novo NMR pulse sequence design using Monte-Carlo optimization techniques. Journal of Magnetic Resonance, 310, 106641

Updated program:

Lapin, J., Nevzorov, A.A. (2020) Computer-Generated Pulse Sequences for 1H-15N and 1Hα-13Cα Separated Local Field Experiments, Journal of Magnetic Resonance, Journal of Magnetic Resonance, 317, 106794

The program was run in an anaconda environment. There are two versions of the program, 1 that is specificially written to split calculations up 
between two GPUs (roulette.py, plotroulette.py), and the other which is written for CPU/Numpy library (roulettenp.py, plotnp.py). It is not recommended 
to run a de novo search on CPU since it is prohibitively slow.

The provided files are as follows:

1. PSLib.py

Useful functions and classes for running the master program

2. roulette.conf

Configuration script for simulation options.

3. polya.xyz

Coordinates of a poly alanine protein segment. Used as input for the specification of the spin system.

4. RNG.inp

Input script for parameter search ranges for every parameter in the pulse sequence to be optimized. See (Lapin, JMR 106641, 2020) for details.

5. sym.inp

Input script for symmetry restraints used in the paramter search. See (Lapin, JMR 106641, 2020) for details.

6. roulette.py

GPU version of the master program for running the MCSA search. Uses python's cupy library to run matrix multiplications on GPU.

7. plotroulette.py

GPU version of the plotting program. Runs using output pulse sequences of roulette.py. Uses python's cupy library. 

8. roulettenp.py

CPU version of the master program. Cannot be used for de novo search due to practical speed limitations.

9. plotnp.py

CPU version of the plotting program. Can run on output from either GPU or CPU version of the master program.

10. createps.py

Python script to turn optimized sequences into TopSpin input for NMR spectrometer.
