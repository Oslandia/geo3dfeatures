Extract geometry features from 3D point clouds

# Content

The project contains the following folders:

+ [geo3dfeatures](./geo3dfeatures) contains source code
+ [docs](./docs) contains some mardown files for documentation purpose and
  images
+ [examples](./examples) contains some Jupyter notebooks for describing data
+ [tests](./tests); `pytest` is used to launch several tests from this folder

Additionally, running the code may generate extra subdirectories in a chosen
data repository (`./data`, by default).

# How to install

This projects runs with Python3, every dependencies are managed
through [poetry](https://poetry.eustace.io/).

## Installation from source

```
$ git clone ssh://git@git.oslandia.net:10022/Oslandia-data/geo3dfeatures.git
$ cd geo3dfeatures
$ virtualenv -p /usr/bin/python3 venv
$ source venv/bin/activate
(venv)$ poetry install
```

# Contribution

See [CONTRIBUTING.md](./CONTRIBUTING.md).

# Run commands

In order to get the available program commands, consider the program help (`geo3d -h`):

```
usage: geo3d [-h] {info,sample,index,featurize,cluster,train,predict} ...

Geo3dfeatures framework for 3D semantic analysis

positional arguments:
  {info,sample,index,featurize,cluster,train,predict}
    info                Describe an input .las file
    sample              Extract a sample of a .las file
    index               Index a point cloud file and serialize it
    featurize           Extract the geometric feature associated to 3D points
    cluster             Cluster a set of 3D points with a k-means algorithm
    train               Train a semantic segmentation model
    predict             Predict 3D point semantic class starting from a
                        trained model

optional arguments:
  -h, --help            show this help message and exit
```

Any further CLI documentation may be printed with `geo3d <command> -h`.

# Documentation

Some documentation is available, that describes the set of considered geometric
features, the fixtures (*i.e.* dummy datasets) used for test purpose and a
practical pipeline use case:

- [Feature set](./docs/features.md)
- [Test fixtures](./docs/test_fixtures.md)
- [Command pipeline](./docs/pipeline.md)

# Examples

The following example has been generated starting from
a [CANUPO](http://nicolas.brodu.net/en/recherche/canupo/) dataset (file `scene.xyz`, with 500k points, 50 neighbors and all the features):

![scene](./docs/images/scene_kmean.png)

___

Oslandia, september 2019
