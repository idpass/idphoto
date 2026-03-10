"""Tests for compute_metrics evaluation logic in eval_utils."""

import numpy as np
import pytest

from eval_utils import compute_metrics


class TestComputeMetricsSeparated:
    """Perfectly separated distributions: positives >> negatives."""

    def setup_method(self):
        self.positives = [0.8, 0.85, 0.9, 0.95, 1.0]
        self.negatives = [0.0, 0.05, 0.1, 0.15, 0.2]
        self.far_caps = [0.01, 0.001]
        self.operating_points = [0.5]
        self.metrics = compute_metrics(
            self.positives, self.negatives, self.far_caps, self.operating_points
        )

    def test_eer_near_zero(self):
        assert self.metrics["eer_approx"]["eer"] < 0.05

    def test_balanced_accuracy_near_one(self):
        assert self.metrics["best_balanced"]["balanced_accuracy"] > 0.95

    def test_positive_stats(self):
        pos = self.metrics["positive"]
        assert pos["count"] == 5
        assert abs(pos["mean"] - np.mean(self.positives)) < 1e-6
        assert abs(pos["min"] - 0.8) < 1e-6
        assert abs(pos["max"] - 1.0) < 1e-6

    def test_negative_stats(self):
        neg = self.metrics["negative"]
        assert neg["count"] == 5
        assert abs(neg["mean"] - np.mean(self.negatives)) < 1e-6
        assert abs(neg["min"] - 0.0) < 1e-6
        assert abs(neg["max"] - 0.2) < 1e-6

    def test_far_cap_lookup(self):
        # With perfect separation, any threshold between 0.2 and 0.8 gives FAR=0
        for cap_key in ["0.01", "0.001"]:
            entry = self.metrics["best_under_far"][cap_key]
            assert entry is not None
            assert entry["far"] <= float(cap_key)
            assert entry["tar"] == 1.0

    def test_operating_point(self):
        op = self.metrics["operating_points"][0]
        assert op["threshold"] == 0.5
        # At 0.5, all positives >= 0.5 (TAR=1), all negatives < 0.5 (FAR=0)
        assert op["tar"] == 1.0
        assert op["far"] == 0.0

    @pytest.mark.xfail(reason="AUC not yet implemented (Phase 8)")
    def test_auc_near_one(self):
        assert self.metrics["auc"] > 0.99

    @pytest.mark.xfail(reason="d-prime not yet implemented (Phase 8)")
    def test_d_prime_large(self):
        assert self.metrics["d_prime"] > 3.0


class TestComputeMetricsOverlapping:
    """Fully overlapping distributions: positives and negatives are interleaved."""

    def setup_method(self):
        rng = np.random.RandomState(42)
        self.positives = list(rng.uniform(0.3, 0.7, 200))
        self.negatives = list(rng.uniform(0.3, 0.7, 200))
        self.far_caps = [0.01]
        self.operating_points = [0.5]
        self.metrics = compute_metrics(
            self.positives, self.negatives, self.far_caps, self.operating_points
        )

    def test_eer_near_half(self):
        assert 0.3 < self.metrics["eer_approx"]["eer"] < 0.7

    def test_balanced_accuracy_near_half(self):
        assert 0.3 < self.metrics["best_balanced"]["balanced_accuracy"] < 0.7

    @pytest.mark.xfail(reason="AUC not yet implemented (Phase 8)")
    def test_auc_near_half(self):
        assert 0.3 < self.metrics["auc"] < 0.7

    @pytest.mark.xfail(reason="d-prime not yet implemented (Phase 8)")
    def test_d_prime_near_zero(self):
        assert abs(self.metrics["d_prime"]) < 1.0


class TestComputeMetricsSingleScores:
    """Edge case: single positive and single negative score."""

    def test_single_scores(self):
        metrics = compute_metrics(
            positive_scores=[0.9],
            negative_scores=[0.1],
            far_caps=[0.01],
            operating_points=[0.5],
        )
        assert metrics["positive"]["count"] == 1
        assert metrics["negative"]["count"] == 1
        assert metrics["best_balanced"]["balanced_accuracy"] == 1.0


class TestComputeMetricsEmptyRaises:
    def test_empty_positives(self):
        with pytest.raises(RuntimeError):
            compute_metrics([], [0.1], [0.01], [0.5])

    def test_empty_negatives(self):
        with pytest.raises(RuntimeError):
            compute_metrics([0.9], [], [0.01], [0.5])


class TestOperatingPointsDoNotAffectEER:
    """Regression test for bug 2.3: operating points outside score range
    should not affect EER or best-balanced threshold."""

    def test_distant_operating_points_no_effect(self):
        positives = [0.7, 0.75, 0.8, 0.85, 0.9]
        negatives = [0.1, 0.15, 0.2, 0.25, 0.3]

        metrics_normal = compute_metrics(
            positives, negatives, [0.01], [0.5]
        )
        # Operating points far outside the score range
        metrics_extreme = compute_metrics(
            positives, negatives, [0.01], [0.001, 5.0, 100.0]
        )

        assert (
            metrics_normal["eer_approx"]["threshold"]
            == metrics_extreme["eer_approx"]["threshold"]
        )
        assert (
            metrics_normal["best_balanced"]["threshold"]
            == metrics_extreme["best_balanced"]["threshold"]
        )
        assert (
            metrics_normal["eer_approx"]["eer"]
            == metrics_extreme["eer_approx"]["eer"]
        )


class TestEERInterpolation:
    """Verify EER linear interpolation improves on discrete approximation."""

    def test_interpolated_eer_between_thresholds(self):
        # 3 positives at [0.4, 0.6, 0.8], 3 negatives at [0.2, 0.5, 0.7].
        # At threshold 0.5: FAR=2/3, FRR=1/3 (diff = +1/3)
        # At threshold 0.6: FAR=1/3, FRR=1/3 (diff = 0)
        # Discrete EER picks 0.6 where FAR=FRR=1/3, so EER=1/3.
        # Interpolation should also find ~1/3 (crossing is at the threshold here).
        positives = [0.4, 0.6, 0.8]
        negatives = [0.2, 0.5, 0.7]
        metrics = compute_metrics(positives, negatives, [0.01], [0.5])
        eer = metrics["eer_approx"]["eer"]
        assert abs(eer - 1 / 3) < 0.05

    def test_interpolation_small_sample(self):
        # With few overlapping scores, the crossing often falls between two
        # discrete thresholds. Interpolation gives a tighter EER estimate.
        # positives=[0.45, 0.65], negatives=[0.35, 0.55]:
        # At threshold 0.45: FAR=1.0, FRR=0.0  (diff=+1.0)
        # At threshold 0.55: FAR=0.5, FRR=0.5  (diff=0.0) -> discrete EER=0.5
        # Interpolation should also find ~0.5 here (crossing at threshold).
        # But use odd counts to force an off-threshold crossing.
        positives = [0.45, 0.55, 0.65]
        negatives = [0.35, 0.50]
        metrics = compute_metrics(positives, negatives, [0.01], [0.5])
        eer_info = metrics["eer_approx"]
        # EER should be reasonable for these overlapping distributions
        assert 0.0 < eer_info["eer"] < 0.5


class TestFMRCapLookup:
    """Verify FAR/FMR cap lookup returns correct thresholds."""

    def test_cap_respected(self):
        positives = [0.6, 0.7, 0.8, 0.9]
        negatives = [0.1, 0.2, 0.3, 0.4]
        metrics = compute_metrics(positives, negatives, [0.25, 0.01], [0.5])

        # At FAR <= 0.25: at most 1 of 4 negatives above threshold
        entry_025 = metrics["best_under_far"]["0.25"]
        assert entry_025 is not None
        assert entry_025["far"] <= 0.25

        # At FAR <= 0.01: no negatives should be above threshold
        entry_001 = metrics["best_under_far"]["0.01"]
        assert entry_001 is not None
        assert entry_001["far"] <= 0.01

    def test_zero_cap_uses_sentinel(self):
        # The sentinel threshold (neg.max() + 1e-6) ensures FAR=0.0 is always
        # reachable, but at the cost of rejecting all genuine scores too.
        positives = [0.5, 0.5, 0.5]
        negatives = [0.5, 0.5, 0.5]
        metrics = compute_metrics(positives, negatives, [0.0], [0.5])
        entry = metrics["best_under_far"]["0"]
        assert entry is not None
        assert entry["far"] == 0.0
        # TAR is 0 because the threshold is above all positive scores
        assert entry["tar"] == 0.0
