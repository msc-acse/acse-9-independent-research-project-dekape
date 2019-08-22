# FullwaveQC

FullwaveQC is a diagnostic package to assist FWI inversions and timelapse implementation with
[Fullwave3D](http://fullwave3d.github.io/) Rev689.

The tools provided by FullwaveQC consist of a series of visualisation and signal processing functions that can be used
to analyse and quality-check the inversion inputs and outputs of Fullwave3D.


## Installation
It is recommended that FullwaveQC is installed using the Conda package manager for the software dependcies. Conda can be
obtained by installing [Anaconda](https://www.anaconda.com/distribution/) for the Python 3.7 version.

FullwaveQC does not support command-line interface and is best used in .py scripts or Jupyter Notebooks.
To install FullwaveQC, including source code, tests and examples notebooks do:
   
 
    # clone repository
    git clone https://github.com/msc-acse/acse-9-independent-research-project-dekape
    cd acse-9-independent-research-project-dekape
    
    # install requirements in python environment
    pip install -r requirements.txt
    
    # install fullwaveqc in python environment
    python setup.py install
    
    # in a python script or Jupyter Notebook import FullwaveQC as
    from fullwaveqc import tools, geom, visual, siganalysis, inversion


## Examples
Full documentation can be found as an html file in 'docs/_build/html/index.html'

### Analysing a seismic dataset
Load, amp, interleave amp, interleave wiggle, xcorr, phasediff

### Analysing an inversion model

### Timelapse

