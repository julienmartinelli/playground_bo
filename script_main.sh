#!/bin/bash

EXP="Zhou Rosenbrock Hartmann"
KERNELS="RBF Matern"
ACQFS="MES UCB EI"
N_REP=20 # need more to have something consistent
BUDGET=30 # same
N_INIT=6 # good rule of thumb is N_INIT=4 * problem_dim
SEED=666
RESULTFOLDER="results"
NAME="test"

mkdir -p $RESULTFOLDER
echo -e "FUNCTIONS=$EXP\nKERNELS=$KERNELS\nACQFS=$ACQFS\nN_REPS=$N_REP\nN_INIT=$N_INIT\nBUDGET=$BUDGET\nSEED=$SEED" > "$RESULTFOLDER/config_$NAME.txt"
python3 scripts/main.py -n $N_REP -ni $N_INIT -b $BUDGET -k $KERNELS -a $ACQFS -e $EXP -se $SEED -s $RESULTFOLDER
