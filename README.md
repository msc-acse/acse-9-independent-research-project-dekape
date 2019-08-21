# FullwaveQC

FullwaveQC is a diagnostic package to assist FWI inversions and timelapse implementation with [Fullwave3D](http://fullwave3d.github.io/)
Rev689.
The tools provided by FullwaveQC consist of a series of visualisation and signal processing functions that can be used
to analyse and quality-check the inversion inputs and outputs of Fullwave3D.


## Installation
It is recommended that FullwaveQC is installed using the Conda package manager for the software dependcies. Conda can be
obtained by installing [Anaconda](https://www.anaconda.com/distribution/) for the Python 3.7 version.

To install FullwaveQC, including source code, tests and examples notebooks do:
init:

    git clone https://github.com/msc-acse/acse-9-independent-research-project-dekape
    cd fullwaveqc
    pip install -r requirements.txt
    

test:
    
    py.test tests


compare to true data --> find observed file from fullwave, guarantees same shapes
check attr for models and data not created by fullwave. can access attr easily to change them
SEGY files only
