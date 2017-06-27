#!/bin/bash

# latent sweep
for L_DIM in `seq 5 5 50`
do
	echo "STARTING RUN FOR LATENT DIMENSIONS $L_DIM at `date`"
	python main.py --arch 500 500 "$L_DIM"
	python run_clustering.py --cluster_train --cluster_test
	echo ""
	echo ""
done

for LA in 500 700 1000
do
	for L_DIM in 10 50
	do
		echo "STARTING RUN FOR TWO HIDDEN DIMENSIONS OF $LA at `date`"
		python main.py --arch $LA $LA $L_DIM
		python run_clustering.py --cluster_train --cluster_test
		echo ""
		echo ""
	done
done


# 3 layers
for LA in 1000 #500 700 1000
do
	for M in 0 #200
	do
		let "LB = $LA - $M"
		for N in 0 #200
		do
			let "LC = $LB - $N"
			for L_DIM in 10 #50 10
			do
				echo "STARTING RUN FOR ARCHITECTURE $LA $LB $LC $L_DIM at `date`"
				python main.py --arch $LA $LB $LC $L_DIM
				python run_clustering.py --cluster_train --cluster_test
				echo ""
				echo ""
			done
		done
	done
done

# 2 layers
for LA in 500 700 1000
do
	for M in 0 #200
	do
		let "LB = $LA - $M"
		for N in 0 #200
		do
			let "LC = $LB - $N"
			for L_DIM in 10 #50 10
			do
				echo "STARTING RUN FOR ARCHITECTURE $LA $LB $LC $L_DIM at `date`"
				python main.py --arch $LA $LB $LC $L_DIM
				python run_clustering.py --cluster_train --cluster_test
				echo ""
				echo ""
			done
		done
	done
done
