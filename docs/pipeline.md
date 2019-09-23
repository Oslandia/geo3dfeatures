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

This program aims at generating small subsets of data starting from a raw
`.las` file.

First we have a 3D point cloud stored as a `.las` file. This dataset may be
literally huge, by containing tens of millions of 3D points. Considering
`./data/input/geocliff.las` the raw input dataset, we may generate a
subsample of 10k 3D points with the `sample` command, as follows:

```
geo3d sample -d data -i geocliff.las -p 10000
```

It will generate a new `.las` file with the 10k points, in
`./data/input/geocliff-10000.las`. This tinier dataset will be far more
practical for testing purpose...


## Index

Once we get a working dataset, we can compute its kd-tree structure, in order
to evaluate distances between points and consequently pre-compute local
neighborhoods. The kd-tree is computed and serialized on the file system with the following command:

```
geo3d index -d data -i geocliff-100000.las -t 1000
```

In such a case, we compute a kd-tree on a sampled file located as
`./data/input/geocliff-100000.las`. We required a kd-tree computing with
1000-leafed branches. We use the input file name for serializing the output at
`data/output/geocliff-100000/kd-tree-leaf-1000.pkl`.

The output file may also be saved at a path specified by the user, for a more
free use (option `--tree-file`).

## Featurize

Once we get the kd-tree structure, we can generate the geometric features that
are associated to the points of the point cloud. The `featurize` program is
focused on geometric feature generation, given a raw 3D point cloud.

```
geo3d featurize -d data -i geocliff-10000.las -n 50 200 -t 1000 -c r g b --chunksize 10000
```

Here we build the point neighborhoods with a kd-tree composed of 1000 points per leaf
(`-t 1000`), each point having 50 and 200 neighbors (`-n 50 200`). The kd-tree file
is recovered thanks to this leaf argument, however we can also specify the tree file
directly (option `--tree-file`).

By default the first three fields of the input data are x, y, z coordinates. As
optional parameters, we can provide some extra fields from the input dataset
such as the raw RGB-color, the density, etc. with the `-c/--extra-columns`
option. You can have `-c r g b` to include the RGB-color for instance.

The feature extraction process is multi-threaded, and point features are
written onto file system by chunks. One may control the writing chunk size with
the `--chunksize` argument.

The output features are stored in a HDF5 file such as
`data/output/geocliff-100000/features/features.h5`. The computed features can be
loaded with the following hdf keys: `/num_0050` and `/num_0200`.

If you have some files which are samples of your complete scene with a
**label** information such as `location_vegetation.las` or
`location_cliff.las`, it's possible to extract some features from them. You
need to pass the `--label-scene` option. By convention, the name of the label
is a suffix of your input file name with a `_` separator. You also need to
compute the kd-tree of the complete scene. The neighborhood look-up must be
carried out in the **complete scene**. The output files will be HDF5 file with
the suffix 'label', e.g. `output/location/features/features_vegetation.h5`.

For instance:

```
geo3d featurize  -d ./workspace -i location_cliff.las -n 50 200 1000 -m 4 -c r g b
```


## Cluster

`cluster` uses k-mean algorithm in order to classify the point of the original
point cloud, starting from the features generated through the `featurize`
command. One may tune the relative importance of each feature during the
clustering process with the help of a configuration file. As an example, one
may run:

```
geo3d cluster -d data -i geocliff-100000.las -n 50 200 -k 2 -c base.ini -p 100
```

This command reads the
`data/output/geocliff-100000/features/features.h5` file, and
computes the corresponding cluster for each of the 10000 points, by following
the feature scheme depicted in `config/base.ini` (*note:* the `-c` argument
must be followed by a filename, not a path). In this configuration, each
feature has an equivalent importance; however some other examples are available
in the `config` folder and home-made configurations may be designed as well by
writing a new `.ini` file.

A parameter `bin` can be set in the configuration file. Its the accumulation
feature bin size, expressed in the same unit than the point cloud. Be careful
with this parameter : it is considered as a floating number in the code.

The k-mean output may be postprocessed in order to make the transitions between
classes smoother and reduce the prediction noise. If such a procedure is
desired, the `-p` parameter must be added to the command, with an integer that
represents the number of neighbors to consider so as to reassess the point
cluster. The underlying postprocessing strategy relies on the computation of a
new kd-tree on the point cloud, and for each point, the computation of the most
frequent clustering label within the neighborhood.

Once the results have been computed, they are stored in
`data/output/geocliff-100000/clustering/kmeans-50-200-base-2-pp100.las`. This
resulting file may be visualized with a 3D data viewer, like CloudCompare. One
may also generate `.xyz` if desired by adding the option `-xyz` to the command.

## Train

As an alternative to clustering, one can train supervised learning algorithms
in order to do 3D point semantic segmentation. This is done through the following command:

```
geo3d train -d data -i Pombourg.las -n 50 200 1000
```

This command considers the dataset provided in `data/input/Pombourg.las`, and
more specifically, every `data/input/Pombourg_<class>.las`, from which
geometric features have been computed (*i.e.* as a pre-requisite, the
`featurize` command had to be run on these sample point clouds) for three
different local neighborhood sizes: 50, 200 and 1000 neighbors.

The `train` command produces as an output a pickelized trained model save in
`data/trained_models/Pombourg.pkl`.

## Predict

The prediction command uses the trained model saved after `train` command
execution, and determines the labels within a 3D point cloud. The chosen model
is logistic regression.

It may be run as follows:

```
geo3d predict -d data -i Pombourg.las -n 50 200 1000 -p 500
```

where `-d data` provides the data folder, `-i Pombourg.las` indicates the
reference point cloud, `-n 50 200 1000` the local neighborhood sizes and `-p
500` the post-processing tuning.

In such a configuration, we will recover the model stored as
`data/trained_models/Pombourg.pkl`, and apply it on
`data/input/Pombourg.las`. The resulting predicted labels are stored in
`data/output/Pombourg/prediction/logreg-50-200-1000-full-pp500.las`.

The whole feature set (except `(x, y, z)` coordinates) is used to train the
model, and as a corollary, to predict labels. This modelling choice is set with
the `full` keyword.

Like in the `cluster` case, one has the possibility to post-process results in
order to mitigate the prediction noise. The `-p` refers to the local
neighborhood size in which the kd-tree is queried, in order to denoise the
label outputs. If the `-p` argument is not specified, the results are saved
without post-processing treatment.
