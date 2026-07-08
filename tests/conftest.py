import os

# Allow PyTorch and energyflow's `wasserstein` OpenMP runtime to coexist in one
# process (macOS aborts with "OMP: Error #15" otherwise). Set before torch is
# imported so the optional energyflow MBR-backend tests can run; harmless when
# wasserstein is absent. Mirrors inference.mbr._import_ef.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.geometry import Geometry


@pytest.fixture(scope="session")
def small_jets():
    return synthetic_matched_dataset(128, seed=0)


@pytest.fixture
def cfg_factory():
    def make(extra=None):
        argv = ["data.n_jets=128", "data.min_val=16", "trainer.batch_size=8"] + (extra or [])
        return load_config(argv)

    return make


@pytest.fixture
def batch(small_jets):
    geom = Geometry()
    ds = MatchedLundDataset(small_jets, geom)
    return collate([ds[i] for i in range(8)]), geom
