#ifndef AUTOSOPHER_H
#define AUTOSOPHER_H

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Autosopher Hyperspace Node Structure
typedef struct Node {
    int key_feature1;
    int key_feature2;
    double data_payload;
    int outcome;           // Discrete outcome (e.g., classification / logic gate)
    int access_count;      // Frequency tracking for noise detection
    struct Node *left;
    struct Node *right;
} Node;

// Tree statistics structure
typedef struct TreeStats {
    int node_count;
    int max_depth;
    size_t memory_bytes;
} TreeStats;

// Core Node Functions
Node* create_node(int f1, int f2, double payload, int outcome);
void free_tree(Node *root);

// Autosophy Learning & Recall
void autosophy_learn(Node **root, int f1, int f2, double payload, int outcome);
int autosophy_recall(Node *root, int f1, int f2, double *out_payload, int *out_outcome);

// Analysis & Stats
void get_tree_stats(Node *root, TreeStats *stats);
void print_tree_structure(Node *root, int level);
void sweep_tree_noise(Node *root, int min_frequency_threshold, int *noise_count, int *total_count);

// Kernels
int kernel_detect_noise(Node *root, int f1, int f2, int min_frequency_threshold);
double kernel_approximate_transcendental(int address_depth);

// Demo Functions
void run_forward_learning_demo(void);
void run_noise_transcendence_demo(void);
void run_interactive_hyperspace(void);

#endif // AUTOSOPHER_H
