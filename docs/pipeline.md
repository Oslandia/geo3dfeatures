# Process pipeline

This document describes the way the different commands must be run in order to
produce point cloud classifications.

## Info

Before any data treatment, one may want to know better the data one targets to
process. By considering an input file `data/input/cloud.las`, some basic
informations are provided with the following command:

```
geo3d info -d data -i cloud.las
```

This command gives the file name, its relative path, its total number of
points, the domain of definition for every the point cloud coordinate (`x`, `y`
and `z`) and the generated output files (kd-tree, features and k-means).

The supported formats are `.las`, `.ply` and `.xyz`. As a side remark, one may
have bonus information with classic terminal commands for `ply` and `xyz` files
as they design text files. For `las` file, one may have full metadata with
`lasinfo` command (through `liblas` package under Linux/Ubuntu).

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
geo3d featurize -d data -i geolithe-extract-10000.las -e democli -n 50 -f full -t 1000 -c r g b -b 1.0 --chunksize 10000
```

Here we build the point neighborhoods with a kd-tree composed of 1000 points
per leaf (`-t 1000`), each point having 50 neighbors (`-n 50`). The kd-tree
file is recovered thanks to this leaf argument, however we can also specify the
tree file directly (option `--tree-file`). We decide to consider all the
geometric features (`-f full`). We do not specify any point quantity (`-p
<nb-points>`), hence all the 10k points are considered. As another argument, we
may specify the accumulation feature bin size (`b`), expressed in the same unit
than the point cloud. Be careful with this parameter : it is considered as a
floating number in the code (if the parameter is set as `-b 1`, the file will
be stored like `-binsize-1.0`).

By default the first three fields of the input data are x, y, z coordinates. As
optional parameters, we can provide some extra fields from the input dataset
such as the raw RGB-color, the density, etc. with the `-c/--extra-columns`
option. You can have `-c r g b` to include the RGB-color for instance.

The feature extraction process is multi-threaded, and point features are
written onto file system by chunks. One may control the writing chunk size with
the `--chunksize` argument.

We call the experiment "democli", for identifying it afterwards. If we do not
give any name, the experiment takes the input file name (here,
"geolithe-extract-10000").

The output features are stored in
`data/output/democli/features/features-n50-full-binsize-1.0.csv`.

## Cluster

As a final command, `cluster` uses k-mean algorithm in order to classify the
point of the original point cloud, starting from the features generated through
the `featurize` command. As an example, one may run:

```
geo3d cluster -d data -e democli -p 10000 -n 50 -f full -k 2 -b 1.0
```

This command reads the
`data/output/democli/features/features-n50-full-binsize-1.0.csv` file, and
computes the corresponding cluster for each of the 10000 points. The results
are stored in
`data/output/democli/clustering/kmeans-n50-full-binsize-1.0-2.xyz`. This
resulting file may be visualized with a 3D data viewer, like CloudCompare. And
voilà!
