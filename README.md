# Entanglement boosting

- Open-source code for _Entanglement boosting: low-volume logical Bell pair preparation for large-scale fault-tolerant quantum computation_.

<img width="2458" height="1129" alt="Screenshot 2025-11-16 at 03 41 04" src="https://github.com/user-attachments/assets/c32e1f24-f4d1-4ecf-87a0-ed034a905478" />


## What is entanglement boosting?

Entanglement boosting protocol prepares a high-fidelity logical Bell pair from noisy physical Bell pairs and additional physical qubits (see figure below). Similar to the _magic-state cultivation_ protocols, boosting begins with error detection with small code (projection onto surface code with variable $d_\mathcal{Bell}$, or `bell-distance`) and rapid escape to a larger code $d_s$ (`surface-distance`), followed by soft-output decoding and postselection based on the soft output (_complementary gap_). 

<img width="1469" height="639" alt="Image" src="https://github.com/user-attachments/assets/ee48a428-eea0-470c-ba47-1b9e8f8c3f0d" />

## Code usage

- Requires `stim` and `pymatching`; no separate installation needed.

- `src/entanglement_boosting.py` provides command line interface with options:

    - `--num-shots` (int, default: 1000)  
    Number of shots (simulation runs) to perform.

    - `--error-probability` (float, default: 0)  
    Error probability for gates in the simulation.

    - `--single-qubit-gate-error-probability` (float, default: None)  
    Error probability specifically for single-qubit gates. Overrides `--error-probability` if set.

    - `--two-qubit-gate-error-probability` (float, default: None)  
    Error probability specifically for two-qubit gates. Overrides `--error-probability` if set.

    - `--reset-error-probability` (float, default: None)  
    Error probability for qubit resets. Overrides `--error-probability` if set.

    - `--measurement-error-probability` (float, default: None)  
    Error probability for measurements. Overrides `--error-probability` if set.

    - `--idle-error-probability` (float, default: None)  
    Error probability for idle qubits. Overrides `--error-probability` if set.

    - `--bell-error-probability` (float, default: 0)  
    Error probability for Bell state preparation.

    - `--parallelism` (int, default: 1)  
    Number of parallel processes to use for the simulation.

    - `--max-shots-per-task` (int, default: 2^20)  
    Maximum number of shots per parallel task.

    - `--bell-distance` (int, default: None)  
    Distance of the Bell pair in the surface code.

    - `--surface-distance` (int, default: 3)  
    Distance parameter for the surface code patch.

    - `--post-selection` (flag)  
    Use early post-selection in simulation (if set).

    - `--print-circuit` (flag)  
    Print the generated circuit (if set).

    - `--dump-results-to` (str, default: None)  
    Path to file to dump simulation results.

    - `--show-progress` (flag)  
    Show progress bar during simulation (if set).

### Example

```sh
python src/entanglement_boosting.py --num-shots 10000 --error-probability 0.001 --bell-error-probability 0.01 --surface-distance 9 --bell-distance 3  --parallelism 4 --show-progress
```

This will run 10,000 shots of the entanglement boosting simulation with surface code distance ($d_s$) of 5, Bell distance ($d_\mathrm{Bell}$) of 3, Bell pair error probability 0.01 and local gate error probability 0.001, using 4 parallel processes and with a progress display.

### Unit testing

```sh
python3 -m unittest *_test.py
```
