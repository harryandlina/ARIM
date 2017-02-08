#!/bin/bash

mainClass=AccureateML.svm.shun.rank.SVMExample

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
		AccureateML.jar \
		$args

		run_time=$(($SECONDS - run_time))

		if [ $run_time -gt 0 ]; then
			break
		fi

		sleep 1
	done
}
initW=-1
step="0"
numFeature=1000
regP=1000
lrN=10
testPath="hdfs:/rcv1.test"
dataPath="hdfs:/rcv1.train"
test100="false"
wPath="rank.resultLargeTest.txt"
minPartN=1

maxRatio=0
ratio=0
ratioStep=1

while [ $ratio -le $maxRatio ]; 
do
    args="$initW $step $numFeature $regP $lrN $testPath $dataPath $test100 $wPath $minPartN"
	do_execute "$args"
	ratio=$((ratio+ratioStep))
done

