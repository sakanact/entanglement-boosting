from __future__ import annotations

import concurrent
import concurrent.futures
import argparse
import enum
import math
import numpy as np
import pymatching
import stim
import sys

import surface_code
import util

from concurrent.futures import ProcessPoolExecutor
from surface_code import SurfaceXSyndromeMeasurement, SurfaceZSyndromeMeasurement, SurfaceSyndromeMeasurement
from util import QubitMapping, Circuit, NoiseConfiguration, MeasurementIdentifier, DetectorIdentifier

FOUR_WEIGHT = surface_code.SurfaceStabilizerPattern.FOUR_WEIGHT
TWO_WEIGHT_DOWN = surface_code.SurfaceStabilizerPattern.TWO_WEIGHT_DOWN
TWO_WEIGHT_UP = surface_code.SurfaceStabilizerPattern.TWO_WEIGHT_UP
TWO_WEIGHT_RIGHT = surface_code.SurfaceStabilizerPattern.TWO_WEIGHT_RIGHT
TWO_WEIGHT_LEFT = surface_code.SurfaceStabilizerPattern.TWO_WEIGHT_LEFT

X_COMPLEMENTARY_GAP_TAG = 'X_COMPLEMENTARY_GAP'
Z_COMPLEMENTARY_GAP_TAG = 'Z_COMPLEMENTARY_GAP'


class SurfaceCodePatch:
    def __init__(self, circuit: Circuit, offset: tuple[int, int], bell_distance: int, distance: int) -> None:
        self.circuit = circuit
        self.offset = offset
        self.bell_distance = bell_distance
        self.distance = distance
        self.syndrome_measurements: dict[tuple[int, int], SurfaceSyndromeMeasurement] = {}
        self._setup_syndrome_measurments()

    def _setup_syndrome_measurments(self) -> None:
        distance = self.distance
        bell_distance = self.bell_distance
        (offset_x, offset_y) = self.offset

        m: SurfaceSyndromeMeasurement
        for i in range(distance):
            for j in range(distance):
                x = offset_x + j * 2
                y = offset_y + i * 2

                satisfied: bool

                # Weight-two syndrome measurements:
                if i == 0 and j % 2 == 0 and j < distance - 1:
                    self._push(SurfaceXSyndromeMeasurement(self.circuit, (x + 1, y - 1), TWO_WEIGHT_DOWN, False))
                if i == distance - 1 and j % 2 == 1:
                    satisfied = (bell_distance != distance)
                    self._push(SurfaceXSyndromeMeasurement(self.circuit, (x + 1, y + 1), TWO_WEIGHT_UP, satisfied))
                if j == 0 and i % 2 == 1:
                    self._push(SurfaceZSyndromeMeasurement(self.circuit, (x - 1, y + 1), TWO_WEIGHT_RIGHT, False))
                if j == distance - 1 and i % 2 == 0 and i < distance - 1:
                    satisfied = (bell_distance != distance)
                    self._push(SurfaceZSyndromeMeasurement(self.circuit, (x + 1, y + 1), TWO_WEIGHT_LEFT, satisfied))

                # Weight-four syndrome measurements:
                if i < distance - 1 and j < distance - 1:
                    if (i + j) % 2 == 0:
                        satisfied = i < j and (i >= bell_distance or j >= bell_distance)
                        m = SurfaceZSyndromeMeasurement(self.circuit, (x + 1, y + 1), FOUR_WEIGHT, satisfied)
                    else:
                        satisfied = j < i and (j >= bell_distance or i >= bell_distance)
                        m = SurfaceXSyndromeMeasurement(self.circuit, (x + 1, y + 1), FOUR_WEIGHT, satisfied)
                    self._push(m)

    def _push(self, m: SurfaceSyndromeMeasurement) -> None:
        self.syndrome_measurements[m.ancilla_position] = m

    def logical_x_pauli_string(self) -> stim.PauliString:
        distance = self.distance
        mapping = self.circuit.mapping
        (offset_x, offset_y) = self.offset

        logical_x: stim.PauliString = stim.PauliString()
        for i in range(distance):
            x = offset_x
            y = offset_y + i * 2
            logical_x *= stim.PauliString('X{}'.format(mapping.get_id(x, y)))
        return logical_x

    def logical_z_pauli_string(self) -> stim.PauliString:
        distance = self.distance
        mapping = self.circuit.mapping
        (offset_x, offset_y) = self.offset

        logical_z: stim.PauliString = stim.PauliString()
        for j in range(distance):
            x = offset_x + j * 2
            y = offset_y
            logical_z *= stim.PauliString('Z{}'.format(mapping.get_id(x, y)))
        return logical_z


class CircuitWithAdditionalProperties:
    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit


def perform_distillation(
        bell_distance: int, surface_distance: int, noise_conf: NoiseConfiguration,
        bell_error_probability: float, post_selection: bool) -> CircuitWithAdditionalProperties:
    SYNDROME_MEASUREMENT_DEPTH = 6
    offset1 = (1, 1)
    offset2 = (1 + surface_distance * 2 + 2, 1)
    (offset1_x, offset1_y) = offset1
    (offset2_x, offset2_y) = offset2

    mapping = QubitMapping(1 + surface_distance * 2 * 2 + 2, 1 + surface_distance * 2)

    patch1_region: list[tuple[int, int]] = [
        (x, y) for x in range(offset1_x - 1, offset2_x - 1) for y in range(mapping.height) if (x + y) % 2 == 0
    ]
    patch2_region: list[tuple[int, int]] = [
        (x, y) for x in range(offset2_x - 1, mapping.width) for y in range(mapping.height) if (x + y) % 2 == 0
    ]
    # Make sure there is no overlap between the two regions.
    for (x, y) in patch1_region:
        assert (x, y) not in patch2_region
    for (x, y) in patch2_region:
        assert (x, y) not in patch1_region

    circuit = Circuit(mapping, noise_conf)
    stim_circuit: stim.Circuit = circuit.circuit

    patch1 = SurfaceCodePatch(circuit, offset1, bell_distance, surface_distance)
    patch2 = SurfaceCodePatch(circuit, offset2, bell_distance, surface_distance)
    m: SurfaceSyndromeMeasurement

    # Make sure that the syndrome measurements on `patch1` and `patch2` are in the same order.
    for (i, m) in enumerate(patch1.syndrome_measurements.values()):
        (x, y) = m.ancilla_position
        peer = list(patch2.syndrome_measurements.values())[i]
        (peer_x, peer_y) = peer.ancilla_position
        assert x - offset1_x == peer_x - offset2_x
        assert y - offset1_x == peer_y - offset2_y

    # First, we share physical Bell pairs and initialize patches with an error-free fashion.
    for i in range(surface_distance):
        for j in range(surface_distance):
            x1 = offset1_x + j * 2
            y1 = offset1_y + i * 2
            if i < bell_distance and j < bell_distance:
                stim_circuit.append('RX', [mapping.get_id(x1, y1)])
            elif i < j:
                stim_circuit.append('R', [mapping.get_id(x1, y1)])
            else:
                stim_circuit.append('RX', [mapping.get_id(x1, y1)])
    for i in range(surface_distance):
        for j in range(surface_distance):
            x2 = offset2_x + j * 2
            y2 = offset2_y + i * 2
            if i < bell_distance and j < bell_distance:
                stim_circuit.append('R', [mapping.get_id(x2, y2)])
            elif i < j:
                stim_circuit.append('R', [mapping.get_id(x2, y2)])
            else:
                stim_circuit.append('RX', [mapping.get_id(x2, y2)])

    for i in range(bell_distance):
        for j in range(bell_distance):
            x1 = offset1_x + j * 2
            y1 = offset1_y + i * 2
            x2 = offset2_x + j * 2
            y2 = offset2_y + i * 2
            stim_circuit.append('CX', [mapping.get_id(x1, y1), mapping.get_id(x2, y2)])

    # Add errors to the patches.
    for i in range(surface_distance):
        for j in range(surface_distance):
            x1 = offset1_x + j * 2
            y1 = offset1_y + i * 2
            x2 = offset2_x + j * 2
            y2 = offset2_y + i * 2
            if i < bell_distance and j < bell_distance:
                # We add noise only to the Bob's endpoints.
                stim_circuit.append('DEPOLARIZE1', [mapping.get_id(x2, y2)], bell_error_probability)
            elif i < j:
                stim_circuit.append('X_ERROR', [mapping.get_id(x1, y1)], noise_conf.reset_error_probability)
                stim_circuit.append('X_ERROR', [mapping.get_id(x2, y2)], noise_conf.reset_error_probability)
            else:
                stim_circuit.append('Z_ERROR', [mapping.get_id(x1, y1)], noise_conf.reset_error_probability)
                stim_circuit.append('Z_ERROR', [mapping.get_id(x2, y2)], noise_conf.reset_error_probability)

    # We perform syndrome measurements on `patch1`.
    # We mark qubits on `patch2` as noise-free during the time.
    circuit.mark_qubits_as_noiseless(patch2_region)

    # Perform `distance = (1 + (distance - 1))` rounds of syndrome measurements on `patch1`.
    for _ in range(SYNDROME_MEASUREMENT_DEPTH):
        for m in patch1.syndrome_measurements.values():
            m.run()
        circuit.place_tick()

    def cast_measurement_id(id: MeasurementIdentifier | None) -> MeasurementIdentifier:
        assert id is not None
        return id

    first_round_measurement_on: dict[tuple[int, int], MeasurementIdentifier] = {
        pos: cast_measurement_id(m.last_measurement) for (pos, m) in patch1.syndrome_measurements.items()
    }
    for _ in range((surface_distance - 1) * SYNDROME_MEASUREMENT_DEPTH):
        for m in patch1.syndrome_measurements.values():
            m.run()
        circuit.place_tick()

    # Perform one round of noise-free syndrome measurements.
    with util.SuppressNoise(circuit):
        for _ in range(SYNDROME_MEASUREMENT_DEPTH):
            for m in patch1.syndrome_measurements.values():
                m.run()
            circuit.place_tick()

    # Let's move on to `patch2`.
    # From now on, the qubits on `patch2` are noisy.
    # We mark qubits on `patch1` as noise-free to allow deferring noise-free logical measurements on `patch1`.
    circuit.mark_qubits_as_noiseless(patch1_region)

    # Set the initial measurement expecations with the first frame measurements on `patch1`.
    for i in range(bell_distance):
        for j in range(bell_distance):
            x1 = offset1_x + j * 2
            y1 = offset1_y + i * 2
            x2 = offset2_x + j * 2
            y2 = offset2_y + i * 2

            def _set_last_measurement(pos1: tuple[int, int], pos2: tuple[int, int]) -> None:
                m = patch2.syndrome_measurements[pos2]
                m.last_measurement = first_round_measurement_on[pos1]
                m.post_selection = post_selection

            if i < bell_distance - 1 and j < bell_distance - 1:
                _set_last_measurement((x1 + 1, y1 + 1), (x2 + 1, y2 + 1))

            if i == 0 and j % 2 == 0 and j < bell_distance - 1:
                _set_last_measurement((x1 + 1, y1 - 1), (x2 + 1, y2 - 1))
            if i == bell_distance - 1 and j % 2 == 1:
                _set_last_measurement((x1 + 1, y1 + 1), (x2 + 1, y2 + 1))
            if j == 0 and i % 2 == 1:
                _set_last_measurement((x1 - 1, y1 + 1), (x2 - 1, y2 + 1))
            if j == bell_distance - 1 and i % 2 == 0 and i < bell_distance - 1:
                _set_last_measurement((x1 + 1, y1 + 1), (x2 + 1, y2 + 1))

    # Perform `distance = (1 + (distance - 1))` rounds of syndrome measurements on `patch2`.
    for _ in range(SYNDROME_MEASUREMENT_DEPTH):
        for m in patch2.syndrome_measurements.values():
            m.run()
        circuit.place_tick()
    for m in patch2.syndrome_measurements.values():
        m.set_post_selection(False)
    for _ in range((surface_distance - 1) * SYNDROME_MEASUREMENT_DEPTH):
        for m in patch2.syndrome_measurements.values():
            m.run()
        circuit.place_tick()

    # Perform one round of noise-free syndrome measurements followed by logical measurements.
    with util.SuppressNoise(circuit):
        for _ in range(SYNDROME_MEASUREMENT_DEPTH):
            for m in patch2.syndrome_measurements.values():
                m.run()
            circuit.place_tick()

        logical_x1 = patch1.logical_x_pauli_string()
        logical_z1 = patch1.logical_z_pauli_string()
        logical_x2 = patch2.logical_x_pauli_string()
        logical_z2 = patch2.logical_z_pauli_string()

        logical_x_measurement = circuit.place_mpp(logical_x1 * logical_x2)
        circuit.place_tick()
        logical_z_measurement = circuit.place_mpp(logical_z1 * logical_z2)

        circuit.place_detector([logical_x_measurement], tag=X_COMPLEMENTARY_GAP_TAG)
        circuit.place_detector([logical_z_measurement], tag=Z_COMPLEMENTARY_GAP_TAG)

        circuit.place_observable_include([logical_x_measurement])
        circuit.place_observable_include([logical_z_measurement])
    return CircuitWithAdditionalProperties(circuit)


class SimulationResultBucket:
    def __init__(self) -> None:
        self.num_valid_samples: int = 0
        self.num_wrong_samples: int = 0

    def append(self, valid: bool) -> None:
        if valid:
            self.num_valid_samples += 1
        else:
            self.num_wrong_samples += 1

    def extend(self, other: SimulationResultBucket):
        self.num_valid_samples += other.num_valid_samples
        self.num_wrong_samples += other.num_wrong_samples

    def __len__(self):
        return self.num_valid_samples + self.num_wrong_samples


class SimulationResults:
    def __init__(self) -> None:
        self.num_discarded_samples: int = 0
        # self.buckets[i] contains the sampling results whose gap are in the range `[i, i + 1)`.`]`
        self.buckets: list[SimulationResultBucket] = [SimulationResultBucket()]

    def append_discarded(self) -> None:
        self.num_discarded_samples += 1

    def append(self, gap: float, valid: bool) -> None:
        gap_floor = int(math.floor(gap))
        if gap_floor >= len(self.buckets):
            self.buckets.extend([SimulationResultBucket() for _ in range(gap_floor + 1 - len(self.buckets))])
            assert len(self.buckets) == gap_floor + 1
        bucket = self.buckets[gap_floor]
        bucket.append(valid)

    def max_gap(self) -> int:
        return len(self.buckets) - 1

    def extend(self, other: SimulationResults):
        self.num_discarded_samples += other.num_discarded_samples
        this_buckets = self.buckets
        other_buckets = other.buckets
        if len(this_buckets) < len(other_buckets):
            this_buckets.extend([SimulationResultBucket() for _ in range(len(other_buckets) - len(this_buckets))])
        elif len(this_buckets) > len(other_buckets):
            other_buckets = other_buckets.copy()
            other_buckets.extend([SimulationResultBucket() for _ in range(len(this_buckets) - len(other_buckets))])

        assert len(this_buckets) == len(other_buckets)
        for i in range(len(this_buckets)):
            this_buckets[i].extend(other_buckets[i])

    def __len__(self):
        return self.num_discarded_samples + sum(len(bucket) for bucket in self.buckets)


def complementary_gap_detectors(dem: stim.DetectorErrorModel, tag: str) -> list[DetectorIdentifier]:
    result: list[DetectorIdentifier] = []
    for i in dem:
        # We don't support stim.DemRepeatBlock.
        assert isinstance(i, stim.DemInstruction)
        if i.type == 'detector' and i.tag == tag:
            targets = i.targets_copy()
            assert len(targets) == 1
            [target] = targets
            assert isinstance(target, stim.DemTarget)
            assert target.is_relative_detector_id()
            result.append(DetectorIdentifier(target.val))
    return result


def perform_simulation(
        circuits: CircuitWithAdditionalProperties,
        num_shots: int,
        seed: int) -> SimulationResults:
    circuit: Circuit = circuits.circuit
    stim_circuit: stim.Circuit = circuit.circuit
    sampler = stim_circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    dem = stim_circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem.without_tags())

    results = SimulationResults()
    postselection_ids = np.array([id.id for id in circuit.detectors_for_post_selection], dtype='uint')

    x_gap_detectors = complementary_gap_detectors(dem, X_COMPLEMENTARY_GAP_TAG)
    z_gap_detectors = complementary_gap_detectors(dem, Z_COMPLEMENTARY_GAP_TAG)
    assert len(x_gap_detectors) == 1
    assert len(z_gap_detectors) == 1
    [x_gap_detector] = x_gap_detectors
    [z_gap_detector] = z_gap_detectors

    for shot in range(num_shots):
        syndrome = detection_events[shot]
        if np.any(syndrome[postselection_ids] != 0):
            results.append_discarded()
            continue

        syndrome[x_gap_detector.id] = 0
        syndrome[z_gap_detector.id] = 0
        p_00, w_00 = matcher.decode(syndrome, return_weight=True)
        prediction = p_00
        min_weight = w_00

        syndrome[x_gap_detector.id] = 0
        syndrome[z_gap_detector.id] = 1
        p_01, w_01 = matcher.decode(syndrome, return_weight=True)
        if w_01 < min_weight:
            prediction = p_01
            min_weight = w_01

        syndrome[x_gap_detector.id] = 1
        syndrome[z_gap_detector.id] = 0
        p_10, w_10 = matcher.decode(syndrome, return_weight=True)
        if w_10 < min_weight:
            prediction = p_10
            min_weight = w_10

        syndrome[x_gap_detector.id] = 1
        syndrome[z_gap_detector.id] = 1
        p_11, w_11 = matcher.decode(syndrome, return_weight=True)
        if w_11 < min_weight:
            prediction = p_11
            min_weight = w_11

        x_gap = abs(min(w_00, w_01) - min(w_10, w_11))
        z_gap = abs(min(w_00, w_10) - min(w_01, w_11))

        gap = min(x_gap, z_gap)
        is_valid = np.array_equal(observable_flips[shot], prediction)

        gap *= 10

        results.append(gap, is_valid)

    return results


def perform_parallel_simulation(
        circuits: CircuitWithAdditionalProperties,
        num_shots: int,
        parallelism: int,
        num_shots_per_task: int,
        show_progress: bool) -> SimulationResults:
    if num_shots / parallelism < 1000 or parallelism == 1:
        return perform_simulation(circuits, num_shots, 0)

    results = SimulationResults()
    progress = 0
    seed = np.random.randint(0, 2 ** 32)
    with ProcessPoolExecutor(max_workers=parallelism) as executor:
        futures: list[concurrent.futures.Future] = []
        remaining_shots = num_shots

        num_shots_per_task = min(num_shots_per_task, (num_shots + parallelism - 1) // parallelism)
        while remaining_shots > 0:
            seed_for_this_task = seed + remaining_shots
            num_shots_for_this_task = min(num_shots_per_task, remaining_shots)
            remaining_shots -= num_shots_for_this_task
            future = executor.submit(perform_simulation, circuits, num_shots_for_this_task, seed_for_this_task)
            futures.append(future)
        try:
            while len(futures) > 0:
                import sys
                if show_progress:
                    print('Progress: {}% ({}/{})\r'.format(
                        round((progress / num_shots) * 100), progress, num_shots), end='')
                concurrent.futures.wait(futures, timeout=None, return_when=concurrent.futures.FIRST_COMPLETED)
                new_futures = []
                for future in futures:
                    if future.done():
                        results.extend(future.result())
                        progress += len(future.result())
                    else:
                        new_futures.append(future)
                futures = new_futures
            if show_progress:
                print()
        finally:
            for future in futures:
                future.cancel()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--num-shots', type=int, default=1000)
    parser.add_argument('--error-probability', type=float, default=0)
    parser.add_argument('--single-qubit-gate-error-probability', type=float, default=None)
    parser.add_argument('--two-qubit-gate-error-probability', type=float, default=None)
    parser.add_argument('--reset-error-probability', type=float, default=None)
    parser.add_argument('--measurement-error-probability', type=float, default=None)
    parser.add_argument('--idle-error-probability', type=float, default=None)
    parser.add_argument('--bell-error-probability', type=float, default=0)
    parser.add_argument('--parallelism', type=int, default=1)
    parser.add_argument('--max-shots-per-task', type=int, default=2 ** 20)
    parser.add_argument('--bell-distance', type=int, default=None)
    parser.add_argument('--surface-distance', type=int, default=3)
    parser.add_argument('--post-selection', action='store_true')
    parser.add_argument('--print-circuit', action='store_true')
    parser.add_argument('--dump-results-to', type=str, default=None)
    parser.add_argument('--show-progress', action='store_true')

    args = parser.parse_args()

    num_shots: int = args.num_shots
    error_probability: float = args.error_probability
    bell_error_probability: float = args.bell_error_probability
    parallelism: int = args.parallelism
    max_shots_per_task: int = args.max_shots_per_task
    surface_distance: int = args.surface_distance
    bell_distance: int = args.bell_distance or surface_distance
    post_selection: bool = args.post_selection
    print_circuit: bool = args.print_circuit
    dump_results_to_filename: str | None = args.dump_results_to
    show_progress: bool = args.show_progress

    single_qubit_gate_error_probability: float
    two_qubit_gate_error_probability: float
    reset_error_probability: float
    measurement_error_probability: float
    idle_error_probability: float
    if args.single_qubit_gate_error_probability is None:
        single_qubit_gate_error_probability = error_probability
    else:
        single_qubit_gate_error_probability = args.single_qubit_gate_error_probability
    if args.two_qubit_gate_error_probability is None:
        two_qubit_gate_error_probability = error_probability
    else:
        two_qubit_gate_error_probability = args.two_qubit_gate_error_probability
    if args.reset_error_probability is None:
        reset_error_probability = error_probability
    else:
        reset_error_probability = args.reset_error_probability
    if args.measurement_error_probability is None:
        measurement_error_probability = error_probability
    else:
        measurement_error_probability = args.measurement_error_probability
    if args.idle_error_probability is None:
        idle_error_probability = error_probability
    else:
        idle_error_probability = args.idle_error_probability

    print('  num-shots = {}'.format(args.num_shots))
    print('  error-probability = {}'.format(args.error_probability))
    print('  single-qubit-gate-error-probability = {}'.format(single_qubit_gate_error_probability))
    print('  two-qubit-gate-error-probability = {}'.format(two_qubit_gate_error_probability))
    print('  reset-error-probability = {}'.format(reset_error_probability))
    print('  measurement-error-probability = {}'.format(measurement_error_probability))
    print('  idle-error-probability = {}'.format(idle_error_probability))
    print('  bell-error-probability = {}'.format(args.bell_error_probability))
    print('  parallelism = {}'.format(args.parallelism))
    print('  max-shots-per-task = {}'.format(args.max_shots_per_task))
    print('  bell-distance = {}'.format(args.bell_distance))
    print('  surface-distance = {}'.format(args.surface_distance))
    print('  full-post-selection = {}'.format(args.post_selection))
    print('  print-circuit = {}'.format(args.print_circuit))
    print('  dump-results-to = {}'.format(args.dump_results_to))
    print('  show-progress = {}'.format(args.show_progress))

    if bell_distance > surface_distance:
        print('bell-distance must be less than or equal to surface-distance.', file=sys.stderr)
        return

    noise_conf = NoiseConfiguration(
        single_qubit_gate_error_probability=single_qubit_gate_error_probability,
        two_qubit_gate_error_probability=two_qubit_gate_error_probability,
        reset_error_probability=reset_error_probability,
        measurement_error_probability=measurement_error_probability,
        idle_error_probability=idle_error_probability)

    circuits: CircuitWithAdditionalProperties = perform_distillation(
        bell_distance, surface_distance, noise_conf, bell_error_probability, post_selection)

    # Assert that the circuit is deterministic.
    _ = circuits.circuit.circuit.detector_error_model()
    # Assert that the circuit is deterministic and has a graph-like DEM.
    _ = circuits.circuit.circuit.detector_error_model(decompose_errors=True)

    if print_circuit:
        print(circuits.circuit.circuit)

    if num_shots == 0:
        return

    results = perform_parallel_simulation(
        circuits, num_shots, parallelism, max_shots_per_task, show_progress)

    if dump_results_to_filename is not None:
        with open(dump_results_to_filename, 'wb') as f:
            import pickle
            pickle.dump(results, f)

    num_unconditionally_discarded = results.num_discarded_samples
    num_total_samples = len(results)
    print('num_total_samples = {}'.format(num_total_samples))

    discard_rates = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for discard_rate in discard_rates:
        num_valid = 0
        num_wrong = 0
        num_discarded = num_unconditionally_discarded
        discarding = True
        gap_threshold = results.max_gap()
        for (gap, bucket) in enumerate(results.buckets):
            if discarding and len(bucket) + num_discarded <= num_total_samples * discard_rate:
                num_discarded += len(bucket)
                continue

            if discarding:
                discarding = False
                gap_threshold = gap

            num_valid += bucket.num_valid_samples
            num_wrong += bucket.num_wrong_samples

        print('Discard rate = {:.2f}, gap threshold = {}, VALID = {}, WRONG = {}, DISCARDED = {}'.format(
            discard_rate, gap_threshold, num_valid, num_wrong, num_discarded))
        print('(VALID + WRONG) / SHOTS = {:.3f}'.format((num_valid + num_wrong) / num_shots))
        print('WRONG / (VALID + WRONG) = {:.3e}'.format(
            math.nan if num_valid + num_wrong == 0 else num_wrong / (num_valid + num_wrong)))
        print()


if __name__ == '__main__':
    main()
