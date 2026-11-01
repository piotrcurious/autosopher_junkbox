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

// Epistemic Status Enum
typedef enum {
    EPISTEMIC_GROUNDED = 0,
    EPISTEMIC_DERIVED = 1,
    EPISTEMIC_HYPOTHETICAL = 2,
    EPISTEMIC_CONTRADICTED = 3,
    EPISTEMIC_UNRESOLVED = 4
} EpistemicStatus;

// Autosopher Hyperspace Node Structure (Self-Balancing AVL Hyperspace Tree)
typedef struct Node {
    int key_feature1;      // Activation key / feature 1
    int key_feature2;      // Activation key / feature 2
    double data_payload;   // Data payload / generative seed / energy
    int outcome;           // Discrete outcome / category class
    int access_count;      // Frequency tracking for noise detection
    int epistemic_status;  // Epistemic status tracking (0: Grounded, 1: Derived, etc.)
    int version;           // Record version for dependency updates
    int deep_belief_code;  // Correlated Deep Belief state / code
    int generative_id;     // Assigned generative template ID
    int height;            // AVL node height for self-balancing
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

// Autosophy Learning & Recall (Self-Balancing AVL Hyperspace)
void autosophy_learn(Node **root, int f1, int f2, double payload, int outcome);
void autosophy_learn_typed(Node **root, int f1, int f2, double payload, int outcome, int status, int version);
int autosophy_recall(Node *root, int f1, int f2, double *out_payload, int *out_outcome);
int autosophy_recall_typed(Node *root, int f1, int f2, double *out_payload, int *out_outcome, int *out_status, int *out_version);

// Range and Associative Nearest Recall (Extending beyond exact match)
int autosophy_range_query(Node *root, int f1_min, int f1_max, int f2_min, int f2_max, int max_results, int out_f1[], int out_f2[], int out_outcome[]);
int autosophy_nearest_recall(Node *root, int f1, int f2, int *best_f1, int *best_f2, int *out_outcome, double *out_distance);

// Deep Belief, Inference Extension & On-The-Fly Autocoding
int autosophy_correlate_deep_belief(Node *root, int f1, int f2, int *out_deep_belief_code, int *out_generative_id, double *out_seed);
int autosophy_autocode_category(Node **root, int f1, int f2, int deep_belief_code, int generative_id, double seed, int outcome);

// Hyperspace Disk Persistence (Serialization / Deserialization)
int autosophy_save_tree(Node *root, const char *filepath);
int autosophy_load_tree(Node **root, const char *filepath);

// Analysis & Stats
void get_tree_stats(Node *root, TreeStats *stats);
void print_tree_structure(Node *root, int level);
void sweep_tree_noise(Node *root, int min_frequency_threshold, int *noise_count, int *total_count);

// Epistemic Update & Propagation Kernels
int kernel_update_epistemic_status(Node *root, int f1, int f2, int new_status, int new_version);
int kernel_detect_noise(Node *root, int f1, int f2, int min_frequency_threshold);
double kernel_approximate_transcendental(int address_depth);

// Demo Functions
void run_forward_learning_demo(void);
void run_noise_transcendence_demo(void);
void run_interactive_hyperspace(void);
void run_hybrid_tf_demo(void);
void run_generative_demo(void);
void run_knowledge_service_demo(void);
void run_comparison_benchmark_demo(void);
void run_language_demo(void);

#endif // AUTOSOPHER_H
