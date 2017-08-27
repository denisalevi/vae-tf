#!/bin/bash

# latent sweep
for L_DIM in `seq 5 5 20`
do
	echo "STARTING RUN FOR LATENT DIMENSIONS $L_DIM at `date`"
	python main.py --arch 500 500 "$L_DIM"
	python run_clustering.py --clust_train_latent --clust_test_latent --save fc_accuracies.txt
	echo ""
	echo ""
done

for L_DIM in `seq 8 1 13`
do
	echo "STARTING RUN FOR LATENT DIMENSIONS $L_DIM at `date`"
	python main.py --arch 500 500 "$L_DIM"
	python run_clustering.py --clust_train_latent --clust_test_latent
	echo ""
	echo ""
done

# 1 layer
for LA in 500 1000 2000
do
	for L_DIM in 10
	do
		echo "STARTING RUN FOR ONE HIDDEN DIMENSION OF $LA at `date`"
		python main.py --arch $LA $L_DIM
		python run_clustering.py --clust_train_latent --clust_test_latent --save fc_accuracies.txt
		echo ""
		echo ""
	done
done

# 2 layers
for LA in 500 1000 2000
do
	for L_DIM in 10
	do
		echo "STARTING RUN FOR TWO HIDDEN DIMENSIONS OF $LA at `date`"
		python main.py --arch $LA $LA $L_DIM
		python run_clustering.py --clust_train_latent --clust_test_latent --save fc_accuracies.txt
		echo ""
		echo ""
	done
done

## 3 layers
#for LA in 1000 #500 700 1000
#do
#	for M in 0 #200
#	do
#		let "LB = $LA - $M"
#		for N in 0 #200
#		do
#			let "LC = $LB - $N"
#			for L_DIM in 10 #50 10
#			do
#				echo "STARTING RUN FOR ARCHITECTURE $LA $LB $LC $L_DIM at `date`"
#				python main.py --arch $LA $LB $LC $L_DIM
#				python run_clustering.py --clust_train_latent --clust_test_latent
#				echo ""
#				echo ""
#			done
#		done
#	done
#done
#
## 2 layers
#for LA in 500 700 1000
#do
#	for M in 0 #200
#	do
#		let "LB = $LA - $M"
#		for N in 0 #200
#		do
#			let "LC = $LB - $N"
#			for L_DIM in 10 #50 10
#			do
#				echo "STARTING RUN FOR ARCHITECTURE $LA $LB $LC $L_DIM at `date`"
#				python main.py --arch $LA $LB $LC $L_DIM
#				python run_clustering.py --clust_train_latent --clust_test_latent
#				echo ""
#				echo ""
#			done
#		done
#	done
#done

# for dot seperated decimals
export LC_NUMERIC="en_US.UTF-8"

for B in `seq 0.1 0.1 2` `seq 2.5 0.5 10` `seq 15 5 50`
do
	echo "STARTING RUN FOR BETA $B at `date`"
	python main.py --arch 500 500 --beta "$B"
	python run_clustering.py --beta "$B" --clust_train_latent --clust_test_latent --log_folder ./log_beta_sweep
	echo ""
	echo ""
done
