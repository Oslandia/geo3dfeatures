"""Index a point cloud scene

For now, just compute a KDTree for x,y,z coordinates of a point cloud scene and serialize it into a file.
"""

import sys
import pickle
from pathlib import Path

import daiquiri

from geo3dfeatures.io import read_xyz, read_las, read_ply
from geo3dfeatures.extract import compute_tree


logger = daiquiri.getLogger(__name__)


def main(opts):
    """Load a point cloud file and compute a kd-tree
    """
    input_path = Path(opts.datapath, "input", opts.input_file)
    if not input_path.is_file():
        logger.error("No such file '%s'.", input_path)
        sys.exit(1)
    logger.info("Load data...")
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
        experiment = opts.input_file.split(".")[0]
        fname = "kd-tree-leaf-{}.pkl".format(leaf_size)
        output_path = Path(opts.datapath, "output", experiment)
        output_path.mkdir(parents=True, exist_ok=True)
        tree_file = output_path / fname
    logger.info("Compute tree...")
    tree = compute_tree(data[:, :3], leaf_size)
    logger.info("Dump tree into %s...", tree_file)
    with open(tree_file, 'wb') as fobj:
        pickle.dump(tree, fobj)
