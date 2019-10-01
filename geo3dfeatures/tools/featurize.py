import sys
import pickle
from pathlib import Path

import daiquiri

import pandas as pd

from geo3dfeatures.io import read_xyz, read_las, read_ply
from geo3dfeatures.extract import extract


logger = daiquiri.getLogger(__name__)


def _check_neighbors_feature_file(h5path, neighbors):
    """Check if the features are already computed in the output h5 file

    according to the number of neighbors.
    """
    neighbors = neighbors[:]  # copy the list
    # if the file does not exists, one computes features for all neighbors list
    if not h5path.exists():
        return neighbors
    with pd.HDFStore(h5path, mode="r") as store:
        # key in the format '/num_xxxx'
        already_computed = [int(x.split("_")[-1]) for x in store.keys()]
        new_neighbors = [x for x in neighbors if x not in already_computed]
        for num_neighbor in neighbors:
            if num_neighbor in already_computed:
                logger.warning("The features won't be computed for this neighborhood (%s) file '%s'.",
                               num_neighbor, h5path.name)
    return new_neighbors


def experiment_folder_name(filename, is_label_scene=False):
    """Return the name of the folder where the output data will be stored.

    - features extraction
    - clustering

    Parameters
    ----------
    fpath: str
        Input scene filename.
    is_label_scene: bool
        Is the file a sample of a complete scene with a '_label' suffix at the end of
        the filename?

    Returns
    -------
    str

    Examples
    --------
    >>> experiment_folder_name('location.las')
    'location'
    >>> experiment_folder_name('location_cliff.las', True)
    'location'
    """
    label_suffix_separator = "_"
    if not is_label_scene:
        return filename.split(".")[0]
    seq = filename.split(label_suffix_separator)
    return label_suffix_separator.join(seq[:-1])


def label_from_file(filename):
    """Return the label from the file name.

    Suppose you have the label name as a suffix of a
    filename. 'location_vegetation.las' will return 'vegetation'.
    """
    label_suffix_separator = "_"
    # remove the extension and split by '_'
    seq = filename.split(".")[0].split(label_suffix_separator)
    return seq[-1]


def main(opts):
    input_path = Path(opts.datapath, "input", opts.input_file)
    if not input_path.is_file():
        logger.error("No such file '%s'.", input_path)
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
            logger.warning(
                "%d extra fields are expected for the provided input data, however you enter %d field names (%s).",
                data.shape[1] - 3, len(opts.extra_columns), opts.extra_columns
            )
            raise ValueError(
                "The given input columns does not match data shape, i.e. x,y,z plus extra columns."
            )

    experiment = experiment_folder_name(opts.input_file, opts.label_scene)
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
        logger.info("Load kd-tree from file %s...", tree_file)
        tree = pickle.load(fobj)

    # the shape must be the same between the data read from file
    # and the data stored in the kd-tree file, except for the '--label-scene'
    # option (where you have a sample of the scene)
    if not opts.label_scene and tree.data.shape[0] != data.shape[0]:
        logger.info(
            "Input data and data stored in the kd-tree "
            "do not have the same length"
        )
        sys.exit(0)

    output_path = Path(opts.datapath, "output", experiment, "features")
    output_path.mkdir(parents=True, exist_ok=True)
    if opts.label_scene:
        hdf5name = "features_" + label_from_file(opts.input_file) + ".h5"
    else:
        hdf5name = "features.h5"  # complete scene
    output_file = output_path / hdf5name

    neighbors = _check_neighbors_feature_file(output_file, opts.neighbors)
    neighbors = list(sorted(neighbors))

    if len(neighbors) == 0:
        logger.info("Features was extracted for all neighboorhoods %s.", opts.neighbors)
        logger.info("Check the file '%s'", output_file)
        logger.info("Exit program")
        sys.exit(0)

    extra_columns = (
        tuple(opts.extra_columns)
        if opts.extra_columns is not None
        else tuple()
        )
    extract(
        data, tree, output_file, neighbors,
        opts.nb_process, extra_columns,
        opts.chunksize
    )
    logger.info("Results in %s", output_file)
