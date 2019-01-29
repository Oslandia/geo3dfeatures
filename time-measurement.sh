#!/bin/bash

set -e
set -u

export PATH=$HOME/.virtualenvs/univers/bin:$PATH

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$here
cd $here

echo "Run time profiler on examples/example.py"
echo "Numbers of evaluated points: $1"
echo "Sizes of neighborhoods: $2"
echo "Feature sets: $3"

for points in $1 # '1000 5000 10000 50000'
do
    for neighbors in $2 # '20 50 100'
    do
	for feature_set in $3 # 'alphabeta eigenvalues full'
	do
	    echo "Run examples/example.py with $points points, $neighbors neighbors and the $feature_set feature set..."
	    _output_file="data/out/features-$points-$neighbors-$feature_set.csv"
	    _profiling_file="data/profiling/profiling-$points-$neighbors-$feature_set"
	    python -m cProfile -o $_profiling_file examples/example.py -i data/scene.xyz --output-file $_output_file -p $points -n $neighbors -t 1000 -f $feature_set
	    echo "**********"
	done
    done
done

