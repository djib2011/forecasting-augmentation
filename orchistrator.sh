#!/bin/bash
python training/combinations_samples_nw.py -n 14000 -c 2
python training/combinations_samples_nw.py -n 140000 -c 2
python training/combinations_samples_nw.py -n 1400000 -c 2

python training/combinations_samples_nw.py -n 14000 -c 3
python training/combinations_samples_nw.py -n 140000 -c 3
python training/combinations_samples_nw.py -n 1400000 -c 3

python training/combinations_samples_nw.py -n 14000 -c 5
python training/combinations_samples_nw.py -n 140000 -c 5
python training/combinations_samples_nw.py -n 1400000 -c 5
