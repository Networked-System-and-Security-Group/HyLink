import pytest
from analyze_subchunk_logs import attach_path_timings


def test_missing_final_feedback_is_not_fabricated_or_shifted():
    rows = [{'iter': 0}, {'iter': 1}]
    attach_path_timings(rows, {0: (7.0, 3.0)})
    assert rows == [{'iter': 0, 'hccl_ms': 7.0, 'pcie_ms': 3.0},
                    {'iter': 1, 'hccl_ms': None, 'pcie_ms': None}]


def test_path_feedback_outside_formal_iterations_is_rejected():
    with pytest.raises(ValueError):
        attach_path_timings([{'iter': 0}], {1: (7.0, 3.0)})
