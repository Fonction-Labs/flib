import pytest
from flib.utils.parallel import ParallelTqdm
from joblib import delayed

def test_parallel_tqdm():
    def dummy_function(x):
        return x * 2

    parallel = ParallelTqdm(n_jobs=2, total_tasks=5)
    results = parallel(delayed(dummy_function)(i) for i in range(5))
    assert results == [0, 2, 4, 6, 8]
