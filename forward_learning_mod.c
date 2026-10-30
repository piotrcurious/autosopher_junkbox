#include "autosopher.h"

// Create a new memory node
Node* create_node(int f1, int f2, double payload, int outcome) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (!newNode) {
        fprintf(stderr, "Error: Memory allocation failed for Node.\n");
        exit(EXIT_FAILURE);
    }
    newNode->key_feature1 = f1;
    newNode->key_feature2 = f2;
    newNode->data_payload = payload;
    newNode->outcome = outcome;
    newNode->access_count = 1;
    newNode->epistemic_status = EPISTEMIC_GROUNDED;
    newNode->version = 1;
    newNode->deep_belief_code = 0;
    newNode->generative_id = 0;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

// Free allocated tree memory
void free_tree(Node *root) {
    if (root == NULL) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

// Autosophy "Self-Assembly" Learning Function
void autosophy_learn(Node **root, int f1, int f2, double payload, int outcome) {
    autosophy_learn_typed(root, f1, f2, payload, outcome, EPISTEMIC_GROUNDED, 1);
}

void autosophy_learn_typed(Node **root, int f1, int f2, double payload, int outcome, int status, int version) {
    if (*root == NULL) {
        *root = create_node(f1, f2, payload, outcome);
        (*root)->epistemic_status = status;
        (*root)->version = version;
        return;
    }

    Node *current = *root;
    while (1) {
        if (current->key_feature1 == f1 && current->key_feature2 == f2) {
            current->data_payload = payload;
            current->outcome = outcome;
            current->access_count++;
            current->epistemic_status = status;
            current->version = version;
            return;
        }

        if (f1 < current->key_feature1 || (f1 == current->key_feature1 && f2 < current->key_feature2)) {
            if (current->left == NULL) {
                current->left = create_node(f1, f2, payload, outcome);
                current->left->epistemic_status = status;
                current->left->version = version;
                break;
            }
            current = current->left;
        } else {
            if (current->right == NULL) {
                current->right = create_node(f1, f2, payload, outcome);
                current->right->epistemic_status = status;
                current->right->version = version;
                break;
            }
            current = current->right;
        }
    }
}

// Deep Belief Correlation & Inference Extension:
// Looks up layer activation keys (f1, f2); if found, returns deep belief code and generative template id.
int autosophy_correlate_deep_belief(Node *root, int f1, int f2, int *out_deep_belief_code, int *out_generative_id, double *out_seed) {
    Node *current = root;
    while (current != NULL) {
        if (current->key_feature1 == f1 && current->key_feature2 == f2) {
            if (out_deep_belief_code) *out_deep_belief_code = current->deep_belief_code;
            if (out_generative_id) *out_generative_id = current->generative_id;
            if (out_seed) *out_seed = current->data_payload;
            return 1; // Category exists! Trigger generative layer reconstruction
        }
        if (f1 < current->key_feature1 || (f1 == current->key_feature1 && f2 < current->key_feature2)) {
            current = current->left;
        } else {
            current = current->right;
        }
    }
    return 0; // Category missing -> requires on-the-fly autocoding
}

// On-The-Fly Autocoding for New Category:
// Instantly self-assembles a new hyperspace node, assigning deep belief code and generative ID dynamically without retraining
int autosophy_autocode_category(Node **root, int f1, int f2, int deep_belief_code, int generative_id, double seed, int outcome) {
    if (*root == NULL) {
        *root = create_node(f1, f2, seed, outcome);
        (*root)->deep_belief_code = deep_belief_code;
        (*root)->generative_id = generative_id;
        return 1;
    }

    Node *current = *root;
    while (1) {
        if (current->key_feature1 == f1 && current->key_feature2 == f2) {
            current->deep_belief_code = deep_belief_code;
            current->generative_id = generative_id;
            current->data_payload = seed;
            current->outcome = outcome;
            current->access_count++;
            return 0; // Updated existing
        }

        if (f1 < current->key_feature1 || (f1 == current->key_feature1 && f2 < current->key_feature2)) {
            if (current->left == NULL) {
                current->left = create_node(f1, f2, seed, outcome);
                current->left->deep_belief_code = deep_belief_code;
                current->left->generative_id = generative_id;
                return 1; // Autocoded new node
            }
            current = current->left;
        } else {
            if (current->right == NULL) {
                current->right = create_node(f1, f2, seed, outcome);
                current->right->deep_belief_code = deep_belief_code;
                current->right->generative_id = generative_id;
                return 1; // Autocoded new node
            }
            current = current->right;
        }
    }
}

// Instant Address-Based Recall
int autosophy_recall(Node *root, int f1, int f2, double *out_payload, int *out_outcome) {
    return autosophy_recall_typed(root, f1, f2, out_payload, out_outcome, NULL, NULL);
}

int autosophy_recall_typed(Node *root, int f1, int f2, double *out_payload, int *out_outcome, int *out_status, int *out_version) {
    Node *current = root;
    while (current != NULL) {
        if (current->key_feature1 == f1 && current->key_feature2 == f2) {
            if (out_payload) *out_payload = current->data_payload;
            if (out_outcome) *out_outcome = current->outcome;
            if (out_status) *out_status = current->epistemic_status;
            if (out_version) *out_version = current->version;
            return 1;
        }
        if (f1 < current->key_feature1 || (f1 == current->key_feature1 && f2 < current->key_feature2)) {
            current = current->left;
        } else {
            current = current->right;
        }
    }
    return 0;
}

// Update Epistemic status & version propagation
int kernel_update_epistemic_status(Node *root, int f1, int f2, int new_status, int new_version) {
    Node *current = root;
    while (current != NULL) {
        if (current->key_feature1 == f1 && current->key_feature2 == f2) {
            current->epistemic_status = new_status;
            current->version = new_version;
            return 1;
        }
        if (f1 < current->key_feature1 || (f1 == current->key_feature1 && f2 < current->key_feature2)) {
            current = current->left;
        } else {
            current = current->right;
        }
    }
    return 0;
}

// Tree Analysis Helper
static void compute_tree_stats_recursive(Node *root, int current_depth, TreeStats *stats) {
    if (root == NULL) return;

    stats->node_count++;
    stats->memory_bytes += sizeof(Node);
    if (current_depth > stats->max_depth) {
        stats->max_depth = current_depth;
    }

    compute_tree_stats_recursive(root->left, current_depth + 1, stats);
    compute_tree_stats_recursive(root->right, current_depth + 1, stats);
}

void get_tree_stats(Node *root, TreeStats *stats) {
    stats->node_count = 0;
    stats->max_depth = 0;
    stats->memory_bytes = 0;
    if (root != NULL) {
        compute_tree_stats_recursive(root, 1, stats);
    }
}

void print_tree_structure(Node *root, int level) {
    if (root == NULL) return;

    print_tree_structure(root->right, level + 1);

    for (int i = 0; i < level; i++) printf("    ");
    printf("[%d,%d] (hits:%d, out:%d, db:%d, gen:%d)\n",
           root->key_feature1, root->key_feature2, root->access_count, root->outcome,
           root->deep_belief_code, root->generative_id);

    print_tree_structure(root->left, level + 1);
}

void sweep_tree_noise(Node *root, int min_frequency_threshold, int *noise_count, int *total_count) {
    if (root == NULL) return;

    (*total_count)++;
    if (root->access_count < min_frequency_threshold) {
        (*noise_count)++;
        printf("Node [%d, %d] (hits: %d) => FLAGGED AS NOISE / TRANSIENT\n",
               root->key_feature1, root->key_feature2, root->access_count);
    } else {
        printf("Node [%d, %d] (hits: %d, db:%d) => VERIFIED KNOWLEDGE NODE\n",
               root->key_feature1, root->key_feature2, root->access_count, root->deep_belief_code);
    }

    sweep_tree_noise(root->left, min_frequency_threshold, noise_count, total_count);
    sweep_tree_noise(root->right, min_frequency_threshold, noise_count, total_count);
}

// Forward Learning Demo Module Function
void run_forward_learning_demo(void) {
    Node *hyperspace_library = NULL;

    printf("\n==================================================\n");
    printf("   DEMO 1: Zero-Epoch Forward Learning Self-Assembly\n");
    printf("==================================================\n");

    printf("\n--- Phase 1: Self-Assembling Logic Gate Patterns (XOR Simulation) ---\n");
    int training_data[4][3] = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };

    for (int i = 0; i < 4; i++) {
        int f1 = training_data[i][0];
        int f2 = training_data[i][1];
        int outcome = training_data[i][2];
        autosophy_learn(&hyperspace_library, f1, f2, 0.0, outcome);
        printf("[Autosopher Growth] Learned pattern [%d, %d] => Outcome: %d\n", f1, f2, outcome);
    }

    TreeStats stats;
    get_tree_stats(hyperspace_library, &stats);
    printf("\n--- Hyperspace Memory Tree Visualization ---\n");
    print_tree_structure(hyperspace_library, 0);
    printf("\nHyperspace Stats: Total Nodes: %d | Depth: %d | Memory: %zu bytes\n",
           stats.node_count, stats.max_depth, stats.memory_bytes);

    printf("\n--- Phase 2: Instant Address-Based Recall (Zero Matrix Multiplication) ---\n");
    for (int i = 0; i < 4; i++) {
        int f1 = training_data[i][0];
        int f2 = training_data[i][1];
        int retrieved_outcome = -1;
        int hit = autosophy_recall(hyperspace_library, f1, f2, NULL, &retrieved_outcome);
        if (hit) {
            printf("Address Query [%d, %d] => SUCCESS: Retrieved Output = %d\n", f1, f2, retrieved_outcome);
        } else {
            printf("Address Query [%d, %d] => UNKNOWN PATTERN\n", f1, f2);
        }
    }

    free_tree(hyperspace_library);
}
