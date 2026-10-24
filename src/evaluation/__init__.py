from .test_set import GEO_TEST_CASES, build_test_dataset, GeoTestCase

def __getattr__(name):
    if name == "run_evaluation":
        from .ragas_eval import run_evaluation
        return run_evaluation
    raise AttributeError(f"module has no attribute {name}")
