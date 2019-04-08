"""Index a point cloud scene

For now, just compute a KDTree for x,y,z coordinates of a point cloud scene and serialize it into a file.
"""

import os
import pickle
from pathlib import Path

from geo3dfeatures.io import (
    xyz as read_xyz,
    las as read_las,
    ply as read_ply,
    )
from geo3dfeatures.extract import compute_tree


def main(opts):
    """Load a point cloud file and compute a kd-tree
    """
    input_path = Path(opts.datapath, "input", opts.input_file)
    print("load data")
    if input_path.suffix == ".xyz":
        data = read_xyz(str(input_path))
    elif input_path.suffix == ".las":
        data = read_las(str(input_path))
    elif input_path.suffix == ".ply":
        data = read_ply(str(input_path))
    else:
        raise ValueError("Wrong file extension, please send xyz or las file.")

    tree_file = opts.tree_file
    leaf_size = opts.kdtree_leafs
    if not tree_file:
        experiment = (
            opts.experiment
            if opts.experiment is not None
            else opts.input_file.split(".")[0]
            )
        fname = "kd-tree-leaf-{}.pkl".format(leaf_size)
        output_path = Path(opts.datapath, "output", experiment)
        os.makedirs(output_path, exist_ok=True)
        tree_file = output_path / fname
    print("compute tree")
    tree = compute_tree(data[:, :3], leaf_size)
    print("dump tree into {}".format(tree_file))
    with open(tree_file, 'wb') as fobj:
        pickle.dump(tree, fobj)
