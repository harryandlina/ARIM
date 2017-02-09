#!/bin/bash

mainClass=AccurateML.svm.shun.hash.SVMExample

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
step="0"
numFeature=1000
inerN=1000
lrN=10
testPath="rcv1.test"
dataPath="rcv1.train"
test100="1,2,3,4,5"
wPath="zip.resultLargeTest50.txt"
minPartN=1

itqbitN=1
itqitN=20
itqrN=10
upBound=100
splitN="2.0"

maxRatio=0
ratio=0
ratioStep=70

while [ $ratio -le $maxRatio ]; 
do
    args="$initW $step $numFeature $inerN $lrN $testPath $dataPath $test100 $wPath $minPartN $itqbitN $itqitN $itqrN $upBound $splitN"
	do_execute "$args"
	ratio=$((ratio+ratioStep))
done

