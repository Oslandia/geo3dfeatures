import sys
import pickle
from pathlib import Path

import daiquiri

from geo3dfeatures.io import (
    xyz as read_xyz,
    las as read_las,
    ply as read_ply,
    )
from geo3dfeatures.extract import extract


logger = daiquiri.getLogger(__name__)


def main(opts):
    input_path = Path(opts.datapath, "input", opts.input_file)
    if not input_path.is_file():
        logger.error("no such file '%s'.", input_path)
        sys.exit(1)
    if input_path.suffix == ".xyz":
        data = read_xyz(str(input_path))
    elif input_path.suffix == ".las":
        data = read_las(str(input_path))
    elif input_path.suffix == ".ply":
        data = read_ply(str(input_path))
    else:
        raise ValueError("Wrong file extension, please send xyz or las file.")

    if opts.extra_columns is not None:
        if len(opts.extra_columns) + 3 != data.shape[1]:
            logger.warning("Number of fields for the input data: '%d' but you ask '%d'.",
                           data.shape[1], len(opts.extra_columns) + 3)
            raise ValueError("The given input columns does not match data shape, i.e. x,y,z plus extra columns.")

    experiment = (
        opts.experiment
        if opts.experiment is not None
        else opts.input_file.split(".")[0]
        )
    tree_file = opts.tree_file
    if not tree_file:
        tree_file = Path(
            opts.datapath, "output", experiment,
            "kd-tree-leaf-" + str(opts.kdtree_leafs) + ".pkl"
        )
        if not tree_file.exists():
            logger.info(
                "No serialized kd-tree with leaf size = %s. Please index your "
                "point cloud with the 'index' command, then use the "
                "--tree-file or the -t/--kdtree-leafs option.",
                opts.kdtree_leafs
            )
            sys.exit(0)

    with open(tree_file, 'rb') as fobj:
        logger.info("Load kd-tree from file...")
        tree = pickle.load(fobj)

    if tree.data.shape[0] != data.shape[0]:
        logger.info(
            "Input data and data stored in the kd-tree "
            "do not have the same length"
        )
        sys.exit(0)

    if opts.neighbors is not None:
        neighborhood = "n" + str(opts.neighbors)
    elif opts.radius is not None:
        neighborhood = "r" + str(opts.radius)
    else:
        raise ValueError(
            "Error in input neighborhood definition: "
            "neighbors and radius arguments can't be both undefined"
            )
    instance = (
        "features-" + neighborhood + "-"
        + str(opts.feature_set) + "-binsize-" + str(opts.bin_size)
        )
    output_path = Path(opts.datapath, "output", experiment, "features")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path, instance + ".csv")

    extra_columns = tuple(opts.extra_columns) if opts.extra_columns is not None else tuple()
    extract(
        data, tree, output_file, opts.neighbors, opts.radius,
        opts.feature_set, opts.nb_process, extra_columns,
        opts.bin_size, opts.chunksize
    )
    logger.info("Results in %s", output_file)
