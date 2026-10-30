# Autosopher Junkbox & Functional Demo Suite

An implementation of Autosophy concepts (self-assembling knowledge trees, content-addressable hyperspace memory, zero-epoch forward learning, frequency-based noise detection, and transcendental continued fraction approximation).

## Overview

Unlike traditional neural networks that rely on backpropagation, weight tuning, and multiple training epochs, **Autosophy** self-assembles knowledge directly into content-addressable hyperspace tree structures.

This demo suite expands the original conceptual sketches into a fully functional C-based demo system.

---

## Features & Demos

1. **Zero-Epoch Forward Learning (`Demo 1`)**:
   - Self-assembles input feature patterns into content-addressable tree nodes without matrix multiplication or epochs.
   - Demonstrates instant address-based recall for non-linear logic relationships (e.g. XOR gate pattern simulation).

2. **Noise Detection & Transcendental Approximation Kernels (`Demo 2`)**:
   - **Kernel 1: Noise & Anomaly Detection**: Evaluates incoming data streams against address frequency tracking to flag transient anomalies versus valid knowledge tips.
   - **Kernel 2: Transcendental Approximation**: Approximates transcendental constants ($\pi$) through recursive address-chained continued fraction expansion trees.

3. **Interactive Hyperspace Memory Explorer (`Demo 3`)**:
   - Allows users to dynamically inject custom feature patterns, perform address queries, inspect hyperspace memory statistics (node count, depth, byte footprint), and view ASCII tree visualizations.

4. **Hybrid TensorFlow + Autosopher Benchmark (`Demo 4`)**:
   - Combines TensorFlow neural perception (feature extraction & quantization on public MNIST benchmark) with Autosopher hyperspace trees for zero-epoch learning, instant address recall, frequency-based noise detection, and incremental zero-epoch knowledge acquisition without neural weight updates.

---

## Building and Running

### Prerequisites
- C Compiler (`gcc` or `clang` supporting C99)
- `make`

### Compilation
To build the complete demo suite and individual standalone executables, run:
```bash
make
```

To clean build artifacts:
```bash
make clean
```

### Running the Demos

#### Interactive Suite CLI
Run the main menu system:
```bash
./autosopher_suite
```

Or run specific demos directly via command-line flags:
```bash
# Run Demo 1 (Forward Learning)
./autosopher_suite --demo1

# Run Demo 2 (Noise Detection & Transcendence)
./autosopher_suite --demo2

# Run Demo 4 (Hybrid TensorFlow + Autosopher Benchmark)
./autosopher_suite --hybrid

# Run all automated demos sequentially
./autosopher_suite --all
```

#### Standalone Executables
The original conceptual entrypoints are also compiled as standalone executables:
```bash
./forward_learning
./noise_and_transcendence
```

---

## Architecture & Code Structure

- `autosopher.h`: Core data structures (`Node`, `TreeStats`), function declarations, and kernel interfaces.
- `forward_learning_mod.c`: Implementation of tree growth, recall, stats calculation, visualizer, and Forward Learning demo.
- `noise_transcendence_mod.c`: Implementation of anomaly detection and transcendental continued fraction kernels.
- `main.c`: Interactive CLI runner and CLI option parser.
- `Makefile`: Build rules for `autosopher_suite`, `forward_learning`, and `noise_and_transcendence`.
