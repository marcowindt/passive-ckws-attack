CKWS Adapted Refined Score Attack
===

M. Dijkslag, M. Damie, F.W. Hahn, A. Peter, _"Passive query-recovery attack against secure conjunctive keyword 
search schemes"_ (ACNS 2022)

This repo is based on a copy of the original Score Attack publicly available at: 
[Refined-score-atk-SSE](https://github.com/MarcT0K/Refined-score-atk-SSE), the original score 
attack is presented in: M. Damie, F. Hahn, A. Peter, _"A Highly Accurate Query-Recovery Attack against Searchable 
Encryption using Non-Indexed Documents"_ (USENIX 2021)

# About
This code aims at simulating the CKWS-adapted-Refined-score-attack against SSE as PoC.

# Getting started
Download the repo and launch `setup.sh`.

Then, run the dataset preprocessor to speed up experiments.
```
python dataset_preprocessor.py
```

To run an attack scenario:
```
python attack_scenario.py
```

NOTE: The co-occurrence matrices generated by the code can be very big, therefore you can gain significant speed-up
if you run large matrix operations on a GPU. The code is written to run in such fashion using the Tensorflow library.

# Result procedures
All the procedures we used to produce our results are in `result_procedures.py`.

# SLURM

In this repo you can find bash scripts that were used to run experiments on a cluster running with [SLURM workload 
scheduler](https://slurm.schedmd.com/).