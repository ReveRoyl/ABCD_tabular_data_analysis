#!/bin/bash

cat subject_list.txt | parallel -j 10 sbatch run_fmriprep_one.sh {}
