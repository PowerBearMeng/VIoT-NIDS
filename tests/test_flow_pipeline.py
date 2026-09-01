from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from data.behavior_composition import (
    fit_behavior_reference,
    score_behavior_composition,
)
from data.flow_builder import canonical_flow, directional_flow, window_coordinates
from data.microbin_features import DirectionalIATAccumulator, MicroBinAccumulator
from data.neural_intensity import aggregate_soft_masses, build_scope_index
from data.pcap_reader import PacketRecord
from data.spatial_context import (
    build_historical_spatial_samples,
    build_spatial_samples,
    initial_historical_state,
)
from models.flow_encoder import FlowAutoencoder
from models.neural_intensity_context import NeuralIntensityContext
from utils.flow_runtime import complementary_masks, deterministic_masks
from utils.metrics import detection_metrics, equal_error_rate
from utils.scaling import EmpiricalUpperTail
from utils.v4_inference import smooth_max


class FlowConstructionTests(unittest.TestCase):
    def test_reverse_packets_share_key_and_flip_direction(self) -> None:
        forward = PacketRecord(1.0, "10.0.0.1", "10.0.0.2", 111, 222, "udp", 60)
        reverse = PacketRecord(1.1, "10.0.0.2", "10.0.0.1", 222, 111, "udp", 70)
        forward_key, forward_direction = canonical_flow(forward)
        reverse_key, reverse_direction = canonical_flow(reverse)
        self.assertEqual(forward_key, reverse_key)
        self.assertEqual((forward_direction, reverse_direction), (0, 1))

    def test_six_microbin_features(self) -> None:
        accumulator = MicroBinAccumulator(2)
        accumulator.add(0, 0, 60)
        accumulator.add(0, 0, 100)
        accumulator.add(0, 1, 70)
        features = accumulator.finalize()
        self.assertEqual(features.shape, (2, 6))
        np.testing.assert_allclose(features[0], [2, 1, 160, 70, 80, 70])
        np.testing.assert_allclose(features[1], np.zeros(6))

    def test_directional_flow_separates_reverse_five_tuple(self) -> None:
        forward = PacketRecord(1.0, "10.0.0.1", "10.0.0.2", 111, 222, "udp", 60)
        reverse = PacketRecord(1.1, "10.0.0.2", "10.0.0.1", 222, 111, "udp", 70)
        forward_key, forward_direction = directional_flow(forward)
        reverse_key, reverse_direction = directional_flow(reverse)
        self.assertNotEqual(forward_key, reverse_key)
        self.assertEqual((forward_direction, reverse_direction), (0, 0))

    def test_epoch_window_matches_tfusion_half_open_bucket(self) -> None:
        index, start, offset = window_coordinates(
            10.0, origin=1.25, window_seconds=3.0, window_alignment="epoch"
        )
        self.assertEqual(index, 3)
        self.assertAlmostEqual(start, 9.0)
        self.assertAlmostEqual(offset, 1.0)
        boundary = window_coordinates(
            12.0, origin=1.25, window_seconds=3.0, window_alignment="epoch"
        )
        self.assertEqual(boundary, (4, 12.0, 0.0))

    def test_directional_iat_features_use_all_six_channels(self) -> None:
        accumulator = DirectionalIATAccumulator(2)
        accumulator.add(0, 0, 60, None)
        accumulator.add(0, 0, 100, 0.010)
        accumulator.add(0, 0, 80, 0.030)
        features = accumulator.finalize()
        self.assertEqual(features.shape, (2, 6))
        np.testing.assert_allclose(
            features[0],
            [3.0, 240.0, 80.0, np.std([60.0, 100.0, 80.0]), 20.0, 10.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(features[1], np.zeros(6))


class SpatialContextTests(unittest.TestCase):
    def test_target_edge_is_excluded_from_context(self) -> None:
        dataset = SimpleNamespace(
            capture_ids=np.asarray(["c", "c", "c"]),
            window_indices=np.asarray([0, 0, 0]),
            endpoint_a_ips=np.asarray(["hub", "hub", "hub"]),
            endpoint_b_ips=np.asarray(["a", "b", "c"]),
        )
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        samples = build_spatial_samples(
            dataset, embeddings, np.zeros(3), np.arange(3), alpha=2.0, min_reliability=0.001
        )
        np.testing.assert_allclose(samples.contexts[0, :2], [0.5, 1.0])
        np.testing.assert_allclose(samples.contexts[0, 2:], [0.0, 0.0])
        np.testing.assert_array_equal(samples.endpoint_context_counts[0], [2, 0])

    def test_historical_context_uses_no_same_window_edge_or_port(self) -> None:
        # No port attributes are supplied: historical context must depend only
        # on capture/window ordering and IP entities/pairs.
        dataset = SimpleNamespace(
            capture_ids=np.asarray(["c", "c", "c", "c"]),
            window_indices=np.asarray([0, 0, 0, 1]),
            endpoint_a_ips=np.asarray(["camera", "camera", "camera", "camera"]),
            endpoint_b_ips=np.asarray(["nvr", "nvr", "nvr", "nvr"]),
        )
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32
        )
        initial = initial_historical_state(np.asarray([0.25, 0.25], dtype=np.float32))
        samples, state = build_historical_spatial_samples(
            dataset,
            embeddings,
            np.zeros(4),
            np.arange(4),
            initial_state=initial,
            alpha=2.0,
            min_reliability=0.001,
            history_beta=1.0,
            state_update_rate=0.1,
            multiplicity_gamma=0.05,
            reset_each_capture=False,
        )
        expected_initial = np.tile([0.25, 0.25, 0.25, 0.25], (3, 1))
        np.testing.assert_allclose(samples.contexts[:3], expected_initial)
        self.assertEqual(len(state["pairs"]), 1)
        self.assertEqual(next(iter(state["pairs"].values()))["observations"], 2)


class FlowV2Tests(unittest.TestCase):
    def test_complementary_masks_cover_each_bin_once(self) -> None:
        masks = complementary_masks(np.asarray(["a", "b"]), 30, 5, 9)
        np.testing.assert_array_equal(masks.sum(axis=0), np.ones((2, 30), dtype=np.int64))

    def test_deterministic_mask_can_force_real_event(self) -> None:
        occupancy = np.zeros((1, 30), dtype=bool)
        occupancy[0, 29] = True
        mask = deterministic_masks(
            np.asarray(["sparse"]), 30, 0.1, 4, occupancy=occupancy, force_occupied=True
        )
        self.assertTrue(mask[0, 29])

    def test_v2_encoder_preserves_expected_shapes(self) -> None:
        model = FlowAutoencoder(
            architecture="v2",
            feature_dim=6,
            hidden_channels=8,
            embedding_dim=5,
            blocks=2,
            kernel_size=3,
            dropout=0.0,
        )
        values = torch.randn(3, 30, 6)
        mask = torch.zeros(3, 30, dtype=torch.bool)
        mask[:, ::5] = True
        reconstruction, embedding = model(values, mask)
        self.assertEqual(tuple(reconstruction.shape), (3, 30, 6))
        self.assertEqual(tuple(embedding.shape), (3, 5))


class FlowV4Tests(unittest.TestCase):
    def test_soft_behavior_mass_is_target_specific_without_mode_id(self) -> None:
        dataset = SimpleNamespace(
            capture_ids=np.asarray(["c", "c", "c"]),
            window_indices=np.asarray([1, 1, 1]),
            endpoint_a_ips=np.asarray(["camera", "camera", "camera"]),
            endpoint_b_ips=np.asarray(["nvr", "nvr", "nvr"]),
        )
        scope = build_scope_index(dataset, np.arange(3))
        assignments = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        pair, entity_a, entity_b = aggregate_soft_masses(assignments, scope)
        expected = np.asarray([np.log(2.0), np.log(2.0), 0.0], dtype=np.float32)
        np.testing.assert_allclose(pair, expected)
        np.testing.assert_allclose(entity_a, expected)
        np.testing.assert_allclose(entity_b, expected)

    def test_neural_intensity_outputs_continuous_soft_assignments(self) -> None:
        model = NeuralIntensityContext(
            embedding_dim=5, hidden_dim=8, latent_channels=3
        )
        embeddings = torch.randn(4, 5)
        assignments = model.assignments(embeddings)
        self.assertEqual(tuple(assignments.shape), (4, 3))
        torch.testing.assert_close(assignments.sum(dim=1), torch.ones(4))
        parameters = model.expected_parameters(embeddings)
        self.assertEqual([tuple(value.shape) for value in parameters], [(4,)] * 4)

    def test_robust_logsumexp_fusion_does_not_cap_large_scores(self) -> None:
        local = np.asarray([0.0, 1.0, 100.0])
        context = np.asarray([0.0, 2.0, 3.0])
        fused = smooth_max([local, context], 0.5)
        self.assertEqual(float(fused[0]), 0.0)
        self.assertGreater(float(fused[2]), 99.0)
        self.assertGreater(float(fused[2]), float(fused[1]))


class BehaviorCompositionTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> tuple[SimpleNamespace, np.ndarray, np.ndarray, np.ndarray]:
        capture_ids: list[str] = []
        windows: list[int] = []
        endpoint_a: list[str] = []
        endpoint_b: list[str] = []
        modes: list[int] = []

        def add(window: int, mode: int, count: int, a: str = "camera", b: str = "nvr") -> None:
            capture_ids.extend(["capture"] * count)
            windows.extend([window] * count)
            endpoint_a.extend([a] * count)
            endpoint_b.extend([b] * count)
            modes.extend([mode] * count)

        # Normal training composition: mode 1 stays exactly at two, while
        # mode 0 has enough variance for a finite reference standard deviation.
        add(0, 0, 1)
        add(0, 1, 2)
        add(1, 0, 1)
        add(1, 1, 2)
        add(2, 0, 2)
        add(2, 1, 2)
        train_stop = len(modes)
        add(3, 0, 8)
        add(3, 1, 2)
        attack_window = np.arange(train_stop, len(modes), dtype=np.int64)
        unknown_start = len(modes)
        add(4, 0, 5, "new-a", "new-b")
        unknown_window = np.arange(unknown_start, len(modes), dtype=np.int64)
        dataset = SimpleNamespace(
            capture_ids=np.asarray(capture_ids),
            window_indices=np.asarray(windows),
            endpoint_a_ips=np.asarray(endpoint_a),
            endpoint_b_ips=np.asarray(endpoint_b),
        )
        return (
            dataset,
            np.asarray(modes, dtype=np.int64),
            np.arange(train_stop, dtype=np.int64),
            np.concatenate([attack_window, unknown_window]),
        )

    def test_flow_reads_only_its_own_mode_deviation(self) -> None:
        dataset, modes, train_indices, test_indices = self._dataset()
        reference = fit_behavior_reference(
            dataset,
            modes,
            train_indices,
            prototype_count=2,
            use_log_count=True,
            epsilon=1e-3,
        )
        scores = score_behavior_composition(
            dataset,
            modes,
            test_indices[:10],
            reference,
            pair_enabled=True,
            entity_enabled=True,
            positive_deviation_only=True,
        )
        mode_zero = modes[test_indices[:10]] == 0
        mode_one = ~mode_zero
        self.assertTrue(np.all(scores.context_deviation[mode_zero] > 0.0))
        np.testing.assert_allclose(scores.context_deviation[mode_one], 0.0)
        np.testing.assert_array_equal(scores.pair_mode_count[mode_zero], 8)
        np.testing.assert_array_equal(scores.pair_mode_count[mode_one], 2)

    def test_unknown_pair_and_entities_are_neutral(self) -> None:
        dataset, modes, train_indices, test_indices = self._dataset()
        reference = fit_behavior_reference(
            dataset,
            modes,
            train_indices,
            prototype_count=2,
            use_log_count=True,
            epsilon=1e-3,
        )
        unknown = test_indices[10:]
        scores = score_behavior_composition(
            dataset,
            modes,
            unknown,
            reference,
            pair_enabled=True,
            entity_enabled=True,
            positive_deviation_only=True,
        )
        np.testing.assert_allclose(scores.context_deviation, 0.0)

    def test_known_scope_uses_zero_counts_from_inactive_train_windows(self) -> None:
        dataset = SimpleNamespace(
            capture_ids=np.asarray(["capture", "capture"]),
            window_indices=np.asarray([0, 1]),
            endpoint_a_ips=np.asarray(["a", "x"]),
            endpoint_b_ips=np.asarray(["b", "y"]),
        )
        reference = fit_behavior_reference(
            dataset,
            np.asarray([0, 0]),
            np.asarray([0, 1]),
            prototype_count=1,
            use_log_count=True,
            epsilon=1e-3,
        )
        key = ("a", "b")
        self.assertEqual(reference.pair_samples[key], 2)
        self.assertAlmostEqual(float(reference.pair_mean[key][0]), np.log(2.0) / 2.0)


class TailCalibrationTests(unittest.TestCase):
    def test_empirical_upper_tail_is_monotone_and_smoothed(self) -> None:
        tail = EmpiricalUpperTail.fit(np.asarray([0.0, 1.0, 2.0]))
        probabilities = tail.probabilities(np.asarray([-1.0, 1.0, 3.0]))
        np.testing.assert_allclose(probabilities, [1.0, 0.75, 0.25])
        evidence = tail.evidence(np.asarray([-1.0, 1.0, 3.0]))
        self.assertTrue(np.all(np.diff(evidence) > 0.0))


class MetricTests(unittest.TestCase):
    def test_perfect_ranking(self) -> None:
        result = detection_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]), 0.5)
        self.assertAlmostEqual(result["AUROC"], 1.0)
        self.assertAlmostEqual(result["AUPRC"], 1.0)
        self.assertAlmostEqual(result["FPR"], 0.0)
        self.assertAlmostEqual(result["TPR"], 1.0)
        self.assertAlmostEqual(result["EER"], 0.0)

    def test_eer_reports_a_realizable_threshold_bracket(self) -> None:
        result = equal_error_rate(
            np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.4, 0.9])
        )
        self.assertGreaterEqual(result["threshold_upper"], result["threshold_lower"])
        self.assertGreaterEqual(result["value"], 0.0)
        self.assertLessEqual(result["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
