"""Extract a subsample of a .las point cloud and save it into another .las file
"""

from pathlib import Path
import sys

import daiquiri
import laspy
import numpy as np
import pandas as pd

from geo3dfeatures import io

logger = daiquiri.getLogger(__name__)


def sample_las(input_path, output_path, nb_points):
    """Sample nb_points within a las file

    Parameters
    ----------
    input_path : Path
    output_path : Path
    nb_points : int
    """
    input_lasfile = laspy.file.File(input_path, mode="r")
    output_lasfile = laspy.file.File(
        output_path, mode="w", header=input_lasfile.header
    )
    input_points = input_lasfile.points
    sample_mask = np.random.choice(
        np.arange(len(input_points)), size=nb_points, replace=False
    )
    output_lasfile.points = input_points[sample_mask]
    input_lasfile.close()
    output_lasfile.close()


def sample_xyz(input_path, output_path, nb_points):
    """Sample nb_points from a xyz file, by using pandas

    Parameters
    ----------
    input_path : Path
    output_path : Path
    nb_points : int
    """
    df = pd.DataFrame(io.read_xyz(input_path)).sample(nb_points)
    df.columns = ("x", "y", "z", "r", "g", "b")
    df = df.astype(dtype={"r": np.uint8, "g": np.uint8, "b": np.uint8})
    df.to_csv(str(output_path), sep=" ", index=False, header=False)


def main(opts):
    logger.info(
        "Sample %s points from file %s...", opts.sample_points, opts.input_file
    )
    input_path = Path(opts.datapath, opts.input_file)
    output_path = Path(
        input_path.parent,
        input_path.stem + "-" + str(opts.sample_points) + input_path.suffix
        )
    if input_path.suffix == ".las":
        sample_las(input_path, output_path, opts.sample_points)
    elif input_path.suffix == ".xyz":
        sample_xyz(input_path, output_path, opts.sample_points)
    else:
        logger.error(
            "Unknown file extension, please provide a las or xyz file."
        )
        sys.exit(1)
