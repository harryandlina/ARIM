#!/bin/bash

mainClass=AccurateML.kmeans.ZFHashKmeans

function do_execute(){
	args=$1
	while true
	do
		run_time=$SECONDS

		spark-submit \
		--class $mainClass \
		--master local[12] \
		--executor-memory 30g \
		--driver-memory 8g \
		AccurateML.jar \
		$args

		run_time=$(($SECONDS - run_time))

		if [ $run_time -gt 1 ]; then
			break
		fi

		sleep 1
	done
}
k=50
itN=100
numFeatures=28102
centerPath="Enron/part-00000"
dataPath="hdfs:/Enron"
test100="1,2,3,4,5"
isSparse="true"
minPartN=4

itqbitN=1
itqitN=20
itqrN=10
upBound=100
splitN=2


maxRatio=10
ratio=10
ratioStep=10

while [ $ratio -le $maxRatio ]; 
do
    args="$k $itN $numFeatures $centerPath $dataPath $test100 $isSparse $minPartN $itqbitN $itqitN $itqrN $upBound $splitN"
	do_execute "$args"
	ratio=$((ratio+ratioStep))
done

