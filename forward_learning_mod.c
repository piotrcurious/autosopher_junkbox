#include "autosopher.h"

// AVL Helper functions
static int max_int(int a, int b) {
    return (a > b) ? a : b;
}

static int node_height(Node *n) {
    return n ? n->height : 0;
}

static void update_height(Node *n) {
    if (n) {
        n->height = 1 + max_int(node_height(n->left), node_height(n->right));
    }
}

static int get_balance(Node *n) {
    return n ? (node_height(n->left) - node_height(n->right)) : 0;
}

static Node* rotate_right(Node *y) {
    Node *x = y->left;
    Node *T2 = x->right;

    x->right = y;
    y->left = T2;

    update_height(y);
    update_height(x);

    return x;
}

static Node* rotate_left(Node *x) {
    Node *y = x->right;
    Node *T2 = y->left;

    y->left = x;
    x->right = T2;

    update_height(x);
    update_height(y);

    return y;
}

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
    newNode->height = 1;
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

// Key comparison helper: returns <0 if key1 < key2, 0 if equal, >0 if key1 > key2
static int compare_keys(int a1, int a2, int b1, int b2) {
    if (a1 != b1) return a1 - b1;
    return a2 - b2;
}

// Recursive self-balancing AVL insertion
static Node* avl_insert(Node *node, int f1, int f2, double payload, int outcome, int status, int version, int db_code, int gen_id) {
    if (node == NULL) {
        Node *newNode = create_node(f1, f2, payload, outcome);
        newNode->epistemic_status = status;
        newNode->version = version;
        newNode->deep_belief_code = db_code;
        newNode->generative_id = gen_id;
        return newNode;
    }

    int cmp = compare_keys(f1, f2, node->key_feature1, node->key_feature2);
    if (cmp == 0) {
        // Address match: update record & payload, increment frequency
        node->data_payload = payload;
        node->outcome = outcome;
        node->access_count++;
        node->epistemic_status = status;
        node->version = version;
        if (db_code != 0) node->deep_belief_code = db_code;
        if (gen_id != 0) node->generative_id = gen_id;
        return node;
    } else if (cmp < 0) {
        node->left = avl_insert(node->left, f1, f2, payload, outcome, status, version, db_code, gen_id);
    } else {
        node->right = avl_insert(node->right, f1, f2, payload, outcome, status, version, db_code, gen_id);
    }

    update_height(node);
    int balance = get_balance(node);

    // Left-Left Case
    if (balance > 1 && compare_keys(f1, f2, node->left->key_feature1, node->left->key_feature2) < 0) {
        return rotate_right(node);
    }
    // Right-Right Case
    if (balance < -1 && compare_keys(f1, f2, node->right->key_feature1, node->right->key_feature2) > 0) {
        return rotate_left(node);
    }
    // Left-Right Case
    if (balance > 1 && compare_keys(f1, f2, node->left->key_feature1, node->left->key_feature2) > 0) {
        node->left = rotate_left(node->left);
        return rotate_right(node);
    }
    // Right-Left Case
    if (balance < -1 && compare_keys(f1, f2, node->right->key_feature1, node->right->key_feature2) < 0) {
        node->right = rotate_right(node->right);
        return rotate_left(node);
    }

    return node;
}

// Autosophy Learning Functions
void autosophy_learn(Node **root, int f1, int f2, double payload, int outcome) {
    autosophy_learn_typed(root, f1, f2, payload, outcome, EPISTEMIC_GROUNDED, 1);
}

void autosophy_learn_typed(Node **root, int f1, int f2, double payload, int outcome, int status, int version) {
    *root = avl_insert(*root, f1, f2, payload, outcome, status, version, 0, 0);
}

// Deep Belief Correlation & Inference Extension
int autosophy_correlate_deep_belief(Node *root, int f1, int f2, int *out_deep_belief_code, int *out_generative_id, double *out_seed) {
    Node *current = root;
    while (current != NULL) {
        int cmp = compare_keys(f1, f2, current->key_feature1, current->key_feature2);
        if (cmp == 0) {
            if (out_deep_belief_code) *out_deep_belief_code = current->deep_belief_code;
            if (out_generative_id) *out_generative_id = current->generative_id;
            if (out_seed) *out_seed = current->data_payload;
            return 1;
        }
        current = (cmp < 0) ? current->left : current->right;
    }
    return 0;
}

// On-The-Fly Autocoding for New Category
int autosophy_autocode_category(Node **root, int f1, int f2, int deep_belief_code, int generative_id, double seed, int outcome) {
    int found_db = 0, found_gen = 0;
    double found_seed = 0.0;
    int exists = autosophy_correlate_deep_belief(*root, f1, f2, &found_db, &found_gen, &found_seed);

    *root = avl_insert(*root, f1, f2, seed, outcome, EPISTEMIC_GROUNDED, 1, deep_belief_code, generative_id);
    return exists ? 0 : 1;
}

// Instant Address-Based Recall
int autosophy_recall(Node *root, int f1, int f2, double *out_payload, int *out_outcome) {
    return autosophy_recall_typed(root, f1, f2, out_payload, out_outcome, NULL, NULL);
}

int autosophy_recall_typed(Node *root, int f1, int f2, double *out_payload, int *out_outcome, int *out_status, int *out_version) {
    Node *current = root;
    while (current != NULL) {
        int cmp = compare_keys(f1, f2, current->key_feature1, current->key_feature2);
        if (cmp == 0) {
            if (out_payload) *out_payload = current->data_payload;
            if (out_outcome) *out_outcome = current->outcome;
            if (out_status) *out_status = current->epistemic_status;
            if (out_version) *out_version = current->version;
            return 1;
        }
        current = (cmp < 0) ? current->left : current->right;
    }
    return 0;
}

// Update Epistemic status & version propagation
int kernel_update_epistemic_status(Node *root, int f1, int f2, int new_status, int new_version) {
    Node *current = root;
    while (current != NULL) {
        int cmp = compare_keys(f1, f2, current->key_feature1, current->key_feature2);
        if (cmp == 0) {
            current->epistemic_status = new_status;
            current->version = new_version;
            return 1;
        }
        current = (cmp < 0) ? current->left : current->right;
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
    printf("[%d,%d] (h:%d, bal:%d, hits:%d, out:%d, db:%d)\n",
           root->key_feature1, root->key_feature2, root->height, get_balance(root),
           root->access_count, root->outcome, root->deep_belief_code);

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

// Hyperspace Disk Persistence
static void save_node_recursive(Node *node, FILE *fp) {
    if (!node) return;
    fprintf(fp, "%d %d %lf %d %d %d %d %d %d\n",
            node->key_feature1, node->key_feature2, node->data_payload,
            node->outcome, node->access_count, node->epistemic_status,
            node->version, node->deep_belief_code, node->generative_id);
    save_node_recursive(node->left, fp);
    save_node_recursive(node->right, fp);
}

int autosophy_save_tree(Node *root, const char *filepath) {
    FILE *fp = fopen(filepath, "w");
    if (!fp) return 0;
    save_node_recursive(root, fp);
    fclose(fp);
    return 1;
}

int autosophy_load_tree(Node **root, const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) return 0;

    int f1, f2, outcome, access, status, ver, db, gen;
    double payload;

    while (fscanf(fp, "%d %d %lf %d %d %d %d %d %d",
                  &f1, &f2, &payload, &outcome, &access, &status, &ver, &db, &gen) == 9) {
        *root = avl_insert(*root, f1, f2, payload, outcome, status, ver, db, gen);
    }
    fclose(fp);
    return 1;
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
    printf("\n--- Hyperspace Memory Tree Visualization (Self-Balancing AVL) ---\n");
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
