#!/bin/bash

# for dot seperated decimals
export LC_NUMERIC="en_US.UTF-8"

for B in `seq 0.1 0.1 2` `seq 2.5 0.5 10` `seq 15 5 50`
do
	echo "STARTING RUN FOR BETA $B at `date`"
	#python main.py --beta "$B"
	python run_clustering.py --beta "$B" --cluster_train --cluster_test --log_folder ./log_beta_sweep
	echo ""
	echo ""
done
