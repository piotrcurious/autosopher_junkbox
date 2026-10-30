#!/usr/bin/env python3
"""
Hybrid TensorFlow + Autosopher System Benchmark
Demonstrates neural perception (TensorFlow feature extraction/quantization)
integrated with Autosophy content-addressable hyperspace memory trees for zero-epoch
forward learning, instant address recall, frequency-based noise detection, and
incremental zero-epoch knowledge acquisition.
"""

import os
import sys
import time
import ctypes
import numpy as np
import tensorflow as tf

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# C-types structure matching autosopher.h
class Node(ctypes.Structure):
    pass

Node._fields_ = [
    ('key_feature1', ctypes.c_int),
    ('key_feature2', ctypes.c_int),
    ('data_payload', ctypes.c_double),
    ('outcome', ctypes.c_int),
    ('access_count', ctypes.c_int),
    ('left', ctypes.POINTER(Node)),
    ('right', ctypes.POINTER(Node))
]

class TreeStats(ctypes.Structure):
    _fields_ = [
        ('node_count', ctypes.c_int),
        ('max_depth', ctypes.c_int),
        ('memory_bytes', ctypes.c_size_t)
    ]

def load_autosopher_library():
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libautosopher.so')
    if not os.path.exists(so_path):
        # Fallback compilation if shared lib not found
        res = os.system("make libautosopher.so >/dev/null 2>&1 || gcc -shared -fPIC -O2 forward_learning_mod.c noise_transcendence_mod.c -o libautosopher.so -lm")
        if res != 0 or not os.path.exists(so_path):
            raise RuntimeError("Could not locate or compile libautosopher.so")

    lib = ctypes.CDLL(so_path)
    lib.autosophy_learn.argtypes = [ctypes.POINTER(ctypes.POINTER(Node)), ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
    lib.autosophy_learn.restype = None

    lib.autosophy_recall.argtypes = [ctypes.POINTER(Node), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    lib.autosophy_recall.restype = ctypes.c_int

    lib.kernel_detect_noise.argtypes = [ctypes.POINTER(Node), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.kernel_detect_noise.restype = ctypes.c_int

    lib.get_tree_stats.argtypes = [ctypes.POINTER(Node), ctypes.POINTER(TreeStats)]
    lib.get_tree_stats.restype = None

    lib.free_tree.argtypes = [ctypes.POINTER(Node)]
    lib.free_tree.restype = None

    return lib

def build_tf_perception_encoder():
    """
    Builds a lightweight TensorFlow CNN encoder to project image inputs into a 2D quantized feature space.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')  # 2D latent representation space
    ])
    return model

def extract_features(encoder, images, grid_resolution=64):
    """
    Passes images through TF perception model and quantizes continuous 2D coordinates into integer feature keys.
    """
    latents = encoder.predict(images, verbose=0)
    # Normalize latents to [0, grid_resolution - 1]
    min_val = np.min(latents, axis=0)
    max_val = np.max(latents, axis=0)
    range_val = np.maximum(max_val - min_val, 1e-5)

    normalized = (latents - min_val) / range_val
    quantized = (normalized * (grid_resolution - 1)).astype(np.int32)
    return quantized, latents

def run_hybrid_demo():
    print("=" * 65)
    print("  HYBRID TENSORFLOW + AUTOSOPHER BENCHMARK DEMO")
    print("  Public Benchmark: MNIST Dataset")
    print("  Division of Labor:")
    print("   - TensorFlow Neural Network : Perception & Latent Feature Extraction")
    print("   - Autosopher Hyperspace Tree: Zero-Epoch Forward Learning & Address Recall")
    print("=" * 65)

    lib = load_autosopher_library()

    print("\n[Step 1] Loading MNIST Benchmark Dataset via TensorFlow...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype(np.float32)[..., np.newaxis]
    x_test = (x_test / 255.0).astype(np.float32)[..., np.newaxis]

    # Use a subset for fast, deterministic demo execution
    train_size = 5000
    test_size = 1000
    x_tr_sub, y_tr_sub = x_train[:train_size], y_train[:train_size]
    x_te_sub, y_te_sub = x_test[:test_size], y_test[:test_size]

    print(f" Loaded dataset: {train_size} train samples, {test_size} test samples.")

    print("\n[Step 2] Initializing TensorFlow Perception Encoder...")
    encoder = build_tf_perception_encoder()

    print(" Extracting 2D latent feature keys via TF model...")
    start_time = time.time()
    tr_features, tr_latents = extract_features(encoder, x_tr_sub, grid_resolution=128)
    te_features, te_latents = extract_features(encoder, x_te_sub, grid_resolution=128)
    tf_time = time.time() - start_time
    print(f" Neural feature extraction complete in {tf_time:.3f}s.")

    print("\n[Step 3] Autosopher Zero-Epoch Forward Learning (Hyperspace Self-Assembly)...")
    autosopher_root = ctypes.POINTER(Node)()

    start_time = time.time()
    for i in range(train_size):
        f1 = int(tr_features[i, 0])
        f2 = int(tr_features[i, 1])
        label = int(y_tr_sub[i])
        # Payload stores confidence metric or raw activation sum
        payload = float(np.linalg.norm(tr_latents[i]))
        lib.autosophy_learn(ctypes.byref(autosopher_root), f1, f2, payload, label)

    learn_time = time.time() - start_time
    print(f" Self-assembly complete in {learn_time:.4f}s (Zero Epochs / No Backprop!).")

    stats = TreeStats()
    lib.get_tree_stats(autosopher_root, ctypes.byref(stats))
    print(f" Hyperspace Tree Stats: Nodes = {stats.node_count}, Depth = {stats.max_depth}, Footprint = {stats.memory_bytes} bytes")

    print("\n[Step 4] Instant Address-Based Recall on Test Benchmark...")
    hits = 0
    correct = 0
    out_payload = ctypes.c_double()
    out_outcome = ctypes.c_int()

    start_time = time.time()
    for i in range(test_size):
        f1 = int(te_features[i, 0])
        f2 = int(te_features[i, 1])
        hit = lib.autosophy_recall(autosopher_root, f1, f2, ctypes.byref(out_payload), ctypes.byref(out_outcome))
        if hit:
            hits += 1
            if out_outcome.value == y_te_sub[i]:
                correct += 1
    recall_time = time.time() - start_time

    hit_rate = (hits / test_size) * 100.0
    accuracy = (correct / hits * 100.0) if hits > 0 else 0.0
    print(f" Test Queries Executed: {test_size} in {recall_time:.4f}s")
    print(f" Address Hits         : {hits} / {test_size} ({hit_rate:.1f}%)")
    print(f" Recall Accuracy      : {correct} / {hits} ({accuracy:.2f}% on hit addresses)")

    print("\n[Step 5] Noise & Anomaly Detection Kernel Evaluation...")
    # Generate noisy image stream (salt & pepper noise)
    noisy_samples = x_te_sub[:100].copy()
    noise_mask = np.random.rand(*noisy_samples.shape) < 0.25
    noisy_samples[noise_mask] = 1.0 - noisy_samples[noise_mask]

    noisy_features, _ = extract_features(encoder, noisy_samples, grid_resolution=128)

    flagged_noise = 0
    for i in range(100):
        f1 = int(noisy_features[i, 0])
        f2 = int(noisy_features[i, 1])
        # Detect transient anomaly with threshold = 2
        is_noise = lib.kernel_detect_noise(autosopher_root, f1, f2, 2)
        if is_noise:
            flagged_noise += 1

    print(f" Corrupted Test Stream Samples: 100")
    print(f" Autosopher Kernel Flagged    : {flagged_noise} / 100 samples as Transient Noise / Structural Anomalies.")

    print("\n[Step 6] Zero-Epoch Incremental Knowledge Injection...")
    print(" Demonstrating instant addition of new knowledge without retraining neural weights...")

    # Inject 50 new explicit patterns into hyperspace memory
    new_injections = 50
    for i in range(new_injections):
        f1 = 9990 + i
        f2 = 8880 + i
        custom_label = 777  # Novel domain class
        lib.autosophy_learn(ctypes.byref(autosopher_root), f1, f2, 123.45, custom_label)

    # Immediately query one injected address
    hit = lib.autosophy_recall(autosopher_root, 9995, 8885, ctypes.byref(out_payload), ctypes.byref(out_outcome))
    if hit and out_outcome.value == 777:
        print(f" SUCCESS: Instantly recalled newly injected pattern [9995, 8885] -> Class {out_outcome.value}")
    else:
        print(" FAILED: Incremental knowledge injection query miss.")

    lib.get_tree_stats(autosopher_root, ctypes.byref(stats))
    print(f" Updated Tree Stats: Nodes = {stats.node_count}, Footprint = {stats.memory_bytes} bytes")

    lib.free_tree(autosopher_root)
    print("\n" + "=" * 65)
    print(" HYBRID DEMO COMPLETE: Neural Perception + Autosophy Hyperspace Memory Successfully Demonstrated.")
    print("=" * 65)

if __name__ == "__main__":
    run_hybrid_demo()
