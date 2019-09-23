from pathlib import Path
import pytest
from typing import NamedTuple

import pandas as pd

from geo3dfeatures import io


_here = Path(__file__).absolute().parent
DATADIR = _here / "data"
PLYFILE = DATADIR / "input" / "b9.ply"
XYZFILE = DATADIR / "input" / "test.xyz"
LASFILE = DATADIR / "input" / "test.las"
CONFIGDIR = _here / "config"


class Point3D(NamedTuple):
    """List of 3D point coordinates
    """
    x: float
    y: float
    z: float


def test_read_xyz():
    """Test the reading of a .xyz file
    """
    fpath = str(XYZFILE)
    data = io.read_xyz(fpath)
    assert data.shape == (10, 6)


def test_write_xyz():
    """Test the writing of a .xyz file: one reads a .xyz file, then re-writes
    it on the file system. One controls that read and written data structures
    are equivalent.
    """
    fpath = str(XYZFILE)
    written_fpath = DATADIR / "input" / "test_write.xyz"
    data = io.read_xyz(fpath)
    df = pd.DataFrame(data, columns=list("xyzrgb"))
    io.write_xyz(df, written_fpath)
    written_data = io.read_xyz(written_fpath)
    assert data.shape == written_data.shape
    written_fpath.unlink()


def test_read_las():
    """Test the reading of a .las file
    """
    fpath = str(LASFILE)
    data = io.read_las(fpath)
    assert data.shape == (10, 6)


def test_write_las():
    """Test the writing of a .las file: one reads a .xyz file, then re-writes
    it on the file system. One controls that read and written data structures
    are equivalent.
    """
    fpath = str(LASFILE)
    written_fpath = DATADIR / "input" / "test_write.las"
    data = io.read_las(fpath)
    df = pd.DataFrame(data, columns=list("xyzrgb"))
    io.write_las(df, LASFILE, written_fpath)
    written_data = io.read_las(written_fpath)
    assert data.shape == written_data.shape
    written_fpath.unlink()


def test_read_ply():
    """Test the reading of a .ply file
    """
    fpath = str(PLYFILE)
    data = io.read_ply(fpath)
    assert data.shape == (22300, 3)


def test_load_features():
    """Test the feature loading process: it must work with full scene, as well
    as with sampled point clouds. It must then works with several neighborhood
    sizes, as well as with only one neighborhood size.
    """
    # datapath, experiment, neighbors, sample=None
    NEIGHBORS = [10, 50, 200]
    N_FEATURES = 19
    # Test for sample == None
    features = io.load_features(DATADIR, "b9", NEIGHBORS)
    assert features.shape == (22300, 3 + len(NEIGHBORS) * N_FEATURES)
    # Test for sample not None
    features = io.load_features(DATADIR, "b9", NEIGHBORS, "foo")
    assert features is None  # 'foo' is not a valid entry
    # Test for different neighbors
    features = io.load_features(DATADIR, "b9", [NEIGHBORS[0]])
    assert features.shape == (22300, 3 + N_FEATURES)


def test_read_config():
    """Test the config reading process: the configuration file must exist and
    contains a "clustering" key.
    """
    UNEXISTING_CONFIG = CONFIGDIR / "foo.ini"
    WRONG_CONFIG = CONFIGDIR / "wrong.ini"
    VALID_CONFIG = CONFIGDIR / "base.ini"
    with pytest.raises(IOError):
        io.read_config(UNEXISTING_CONFIG)
    with pytest.raises(ValueError):
        io.read_config(WRONG_CONFIG)
    config = io.read_config(VALID_CONFIG)
    assert list(config.keys()) == ["DEFAULT", "clustering"]


def test_instance():
    """Test the instance suffixing with respect to neighborhood definition
    (neighbor amount or radius).
    """
    with pytest.raises(ValueError):
        io.instance(None, None)
    inst = io.instance([10, 50, 200], None)
    assert inst == "10-50-200"
    inst = io.instance(None, 1.0)
    assert inst == "r1.0"
