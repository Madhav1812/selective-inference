#!/bin/bash

DIR=/scratch/PI/sabatti/test
NGENES=100
YSEED=0

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate 

# python sim_ciseqtl.py generateX -o $DIR -g $NGENES -k 10
# python sim_ciseqtl.py generateY -o $DIR -g $NGENES -k 10 -s $YSEED

# python sim_ciseqtl.py runSimes -o $DIR -g $NGENES -k 10 -s $YSEED
# python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED
DIR=/scratch/PI/sabatti/controlled_access_data/cisEQTLS_sim_exp2
YSEED=3


for YSEED in {1..10}
do  
    python sim_ciseqtl.py evalSimes -o $DIR -s $YSEED
done
