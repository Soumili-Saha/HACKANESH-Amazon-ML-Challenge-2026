import os
import sys
from dataclasses import dataclass


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _env(name, default):
    return os.path.expanduser(os.environ.get(name, default))


@dataclass
class Paths:
    code_dir: str = _env("ER_CODE_DIR", os.path.join(ROOT, "src", "pipeline"))
    data_dir: str = _env("ER_DATA_DIR", os.path.join(ROOT, "dataset"))
    parquet_dir: str = _env("ER_PARQUET_DIR", os.path.join(ROOT, "dataset", "parquet"))
    work_dir: str = _env("ER_WORK_DIR", os.path.join(ROOT, "work"))
    out_dir: str = _env("ER_OUT_DIR", os.path.join(ROOT, "runs"))
    submission_dir: str = _env("ER_SUBMISSION_DIR", os.path.join(ROOT, "..", "..", "output"))
    frac: float = 0.15
    seed: int = 42
    n_folds: int = 3
    hold: int = 2

    @property
    def run_dir(self):
        return f"{self.work_dir}/run_frac{self.frac}_seed{self.seed}"

    @property
    def v4_dir(self):
        return f"{self.run_dir}/v4"

    @property
    def final_dir(self):
        return f"{self.work_dir}/final_v4"

    @property
    def test_dir(self):
        return f"{self.work_dir}/test_run_v4"

    def out(self, split, name=""):
        d = os.path.join(self.out_dir, split)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name) if name else d


def load_er(paths):
    if paths.code_dir not in sys.path:
        sys.path.insert(0, paths.code_dir)
    import er_pipeline as er
    return er
