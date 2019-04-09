# Process pipeline

This document describes the way the different commands must be run in order to
produce point cloud classifications.

## Sample

First we have a 3D point cloud stored as a `.las` file. This dataset may be
literally huge, by containing tens of millions of 3D points. Considering
`./data/input/geolithe-extract.las` the raw input dataset, we may generate a
subsample of 10k 3D points with the `sample` command, as follows:

```
geo3d sample -d data -i geolithe-extract.las -p 10000
```

It will generate a new `.las` file with the 10k points, in
`./data/input/geolithe-extract-10000.las`. This tinier dataset will be far more
practical for testing purpose...

## Index

Once we get a working dataset, we can compute its kd-tree structure, in order
to evaluate distances between points and consequently pre-compute local
neighborhoods. The kd-tree is computed and serialized on the file system with the following command:

```
geo3d index -d data -i geocliff-100000.las -e democli -t 1000
```

In such a case, we compute a kd-tree on a sampled file located as
`./data/input/geocliff-100000.las`. We required a kd-tree computing with
1000-leafed branches. The experiment name (`-e democli`) is optional: it is
used for saving the kd-tree in the file system, knowing that we use the input
file name if no experiment name is provided. Here it will be serialized at
`data/output/democli/kd-tree-leaf-1000.pkl`. The output file may also be saved
at a path specified by the user, for a more free use (option `--tree-file`).

## Featurize

Once we get the kd-tree structure, we can generate the geometric features that
are associated to the points of the point cloud:

```
geo3d featurize -d data -i geolithe-extract-10000.las -e democli -n 50 -f full -t 1000 -c r g b
```

Here we build the point neighborhoods with a kd-tree composed of 1000 points
per leaf (`-t 1000`), each point having 50 neighbors (`-n 50`). The kd-tree
file is recovered thanks to this leaf argument, however we can also specify the
tree file directly (option `--tree-file`). We decide to consider all the
geometric features (`-f full`). We do not specify any point quantity (`-p
<nb-points>`), hence all the 10k points are considered.

By default the first three fields of the input data are x, y, z coordinates. As
optional parameters, we can provide some extra fields from the input dataset such as
the raw RGB-color, the density, etc. with the `-c/--extra-columns` option. You can
have `-c r g b` to include the RGB-color for instance.

We call the experiment "democli", for identifying it afterwards. If we do not give
any name, the experiment takes the input file name (here, "geolithe-extract-10000").

The output features are stored in
`data/output/democli/features/features-10000-50-full.csv`.

## Profile

As an alternative to the previous step, one may need to compute the featurize
program running time. A `pstats` utility that wraps the program is written in
`./time-measurement.sh`.

The feature generation may be done as follows:

```
./time-measurement.sh democli 10000 50 full 'x y z r g b' geolithe-extract-10000.las
```

This command generate the `.csv` file that contains features **and** a
profiling file `data/output/democli/profiling/profiling-10000-50-full` that
contains time measurements.

In order to exploit these time measurements, the `profile` command converts profiling files into human-readable files. Hence:

```
geo3d profile -F csv -e democli
```

reads every single file in `data/output/democli/profiling/` folder, and write `.csv` versions in `data/output/democli/timers/`.

## Cluster

As a final command, `cluster` uses k-mean algorithm in order to classify the
point of the original point cloud, starting from the features generated through
the `featurize` command. As an example, one may run:

```
geo3d cluster -d data -e democli -p 10000 -n 50 -f full -k 2
```

This command reads the `data/output/democli/features/feature-10000-50-full.csv`
file, and computes the corresponding cluster for each of the 10000 points. The
results are stored in
`data/output/democli/clustering/kmeans-10000-50-full.xyz`. This resulting file
may be visualized with a 3D data viewer, like CloudCompare. And voilà!
