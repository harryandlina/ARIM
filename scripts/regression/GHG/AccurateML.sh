#!/bin/bash

mainClass=AccurateML.nonLinearRegression.ZFNNLSHPart3

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

		if [ $run_time -gt 0 ]; then
			break
		fi

		sleep 1
	done
}
initW=-1
step="0.01"
numFeature=5232
hiddenN=5
lrN=100
testPath="hdfs:/GHG.test"
dataPath="hdfs:/GHG.train"
test100="1,2,3,4,5"
wPath="nnweights"
isSparse="false"

itqbitN=1
itqitN=20
itqrN=100
minPart=1
upBound=30
splitN="2.0"

maxRatio=1
ratio=1
ratioStep=4

while [ $ratio -le $maxRatio ]; 
do
    args="$initW $step $numFeature $hiddenN $lrN $testPath $dataPath $test100 $wPath $isSparse $itqbitN $itqitN $itqrN $minPart $upBound $splitN"
	do_execute "$args"
	ratio=$((ratio+ratioStep))
done

