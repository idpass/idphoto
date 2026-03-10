"""Tests for shared evaluation helper functions in eval_utils."""

import numpy as np
import pytest

from eval_utils import (
    cosine_similarity,
    l2_distance,
    l2_normalize,
    parse_budgets,
    parse_float_list,
    parse_providers,
)


class TestL2Normalize:
    def test_normalizes_vector(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_preserves_direction(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = l2_normalize(v)
        assert result[0] > 0 and result[1] > 0
        assert abs(result[0] / result[1] - 3.0 / 4.0) < 1e-6

    def test_near_zero_norm_raises(self):
        v = np.array([1e-20, 1e-20], dtype=np.float64)
        with pytest.raises(ValueError, match="Near-zero-norm"):
            l2_normalize(v)

    def test_exact_zero_raises(self):
        v = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="Near-zero-norm"):
            l2_normalize(v)

    def test_already_normalized_is_idempotent(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = l2_normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-7)

    def test_negative_values(self):
        v = np.array([-3.0, 4.0], dtype=np.float32)
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        assert result[0] < 0


class TestCosineSimilarity:
    def test_parallel_vectors(self):
        a = l2_normalize(np.array([1.0, 2.0, 3.0]))
        b = l2_normalize(np.array([2.0, 4.0, 6.0]))
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = l2_normalize(np.array([1.0, 0.0]))
        b = l2_normalize(np.array([0.0, 1.0]))
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_antiparallel_vectors(self):
        a = l2_normalize(np.array([1.0, 0.0]))
        b = l2_normalize(np.array([-1.0, 0.0]))
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


class TestL2Distance:
    def test_same_vector_is_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(l2_distance(v, v)) < 1e-10

    def test_known_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(l2_distance(a, b) - 5.0) < 1e-6


class TestParseFloatList:
    def test_empty_string(self):
        assert parse_float_list("") == []
        assert parse_float_list("   ") == []

    def test_single_value(self):
        assert parse_float_list("0.5") == [0.5]

    def test_multiple_values(self):
        assert parse_float_list("0.1,0.2,0.3") == [0.1, 0.2, 0.3]

    def test_whitespace_handling(self):
        assert parse_float_list("  0.1 , 0.2 , 0.3  ") == [0.1, 0.2, 0.3]

    def test_trailing_comma(self):
        assert parse_float_list("0.1,0.2,") == [0.1, 0.2]


class TestParseBudgets:
    def test_empty_string(self):
        assert parse_budgets("") == []

    def test_single_value(self):
        assert parse_budgets("1024") == [1024]

    def test_multiple_values(self):
        assert parse_budgets("2048,1536,1024") == [2048, 1536, 1024]

    def test_whitespace_handling(self):
        assert parse_budgets("  2048 , 1024 ") == [2048, 1024]

    def test_negative_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_budgets("-1")

    def test_zero_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_budgets("0")


class TestParseProviders:
    def test_single_provider(self):
        assert parse_providers("CPUExecutionProvider") == ["CPUExecutionProvider"]

    def test_multiple_providers(self):
        result = parse_providers("CUDAExecutionProvider,CPUExecutionProvider")
        assert result == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_providers("")
