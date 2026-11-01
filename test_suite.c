#include "autosopher.h"
#include <assert.h>

static void test_avl_balancing(void) {
    printf("[Test 1] AVL Self-Balancing Insertion & Height Invariance...\n");
    Node *root = NULL;

    // Insert 100 strictly ascending keys (which would cause depth 100 in unbalanced BST)
    for (int i = 0; i < 100; i++) {
        autosophy_learn(&root, i, i, (double)i, i % 10);
    }

    TreeStats stats;
    get_tree_stats(root, &stats);

    assert(stats.node_count == 100);
    // For AVL tree with 100 nodes, max theoretical height is < 1.44 * log2(100) ~ 9-10
    assert(stats.max_depth <= 10);
    printf(" -> PASS: 100 ascending insertions balanced to max depth %d (<= 10)\n", stats.max_depth);

    free_tree(root);
}

static void test_recall_and_nearest(void) {
    printf("[Test 2] Exact Recall & Nearest-Address Associative Recall...\n");
    Node *root = NULL;

    autosophy_learn(&root, 10, 20, 3.14, 1);
    autosophy_learn(&root, 30, 40, 2.71, 2);
    autosophy_learn(&root, 50, 60, 1.41, 3);

    double payload = 0.0;
    int outcome = -1;

    // Exact recall
    int hit = autosophy_recall(root, 30, 40, &payload, &outcome);
    assert(hit == 1);
    assert(outcome == 2);
    assert(fabs(payload - 2.71) < 1e-4);

    // Nearest recall on non-existent address [29, 39]
    int best_f1, best_f2, best_out;
    double dist;
    hit = autosophy_nearest_recall(root, 29, 39, &best_f1, &best_f2, &best_out, &dist);
    assert(hit == 1);
    assert(best_f1 == 30 && best_f2 == 40 && best_out == 2);
    assert(dist < 2.0);
    printf(" -> PASS: Exact match and nearest associative recall verified.\n");

    free_tree(root);
}

static void test_range_query(void) {
    printf("[Test 3] Multi-Dimensional Hyperspace Range Query...\n");
    Node *root = NULL;

    for (int i = 0; i < 50; i++) {
        autosophy_learn(&root, i, i * 2, (double)i, i % 5);
    }

    int out_f1[50], out_f2[50], out_outcome[50];
    int count = autosophy_range_query(root, 10, 20, 20, 40, 50, out_f1, out_f2, out_outcome);
    assert(count == 11); // items from 10 to 20
    for (int i = 0; i < count; i++) {
        assert(out_f1[i] >= 10 && out_f1[i] <= 20);
        assert(out_f2[i] >= 20 && out_f2[i] <= 40);
    }
    printf(" -> PASS: Range query retrieved %d matching points within bounding box.\n", count);

    free_tree(root);
}

static void test_epistemic_updates(void) {
    printf("[Test 4] Epistemic Status & Version Propagation...\n");
    Node *root = NULL;

    autosophy_learn_typed(&root, 5, 5, 10.0, 1, EPISTEMIC_GROUNDED, 1);

    double p = 0.0;
    int o = -1, status = -1, ver = -1;
    autosophy_recall_typed(root, 5, 5, &p, &o, &status, &ver);
    assert(status == EPISTEMIC_GROUNDED && ver == 1);

    // Update status to CONTRADICTED and increment version
    int res = kernel_update_epistemic_status(root, 5, 5, EPISTEMIC_CONTRADICTED, 2);
    assert(res == 1);

    autosophy_recall_typed(root, 5, 5, &p, &o, &status, &ver);
    assert(status == EPISTEMIC_CONTRADICTED && ver == 2);
    printf(" -> PASS: Epistemic status transitions and version increments verified.\n");

    free_tree(root);
}

static void test_deep_belief_and_autocoding(void) {
    printf("[Test 5] Deep Belief Correlation & On-The-Fly Autocoding...\n");
    Node *root = NULL;

    int db_code = 0, gen_id = 0;
    double seed = 0.0;

    // Novel pattern -> should autocode
    int autocoded = autosophy_autocode_category(&root, 77, 88, 7788, 42, 0.99, 42);
    assert(autocoded == 1);

    // Existing pattern -> should now correlate
    int found = autosophy_correlate_deep_belief(root, 77, 88, &db_code, &gen_id, &seed);
    assert(found == 1);
    assert(db_code == 7788 && gen_id == 42);
    assert(fabs(seed - 0.99) < 1e-4);
    printf(" -> PASS: On-the-fly autocoding and deep belief correlation verified.\n");

    free_tree(root);
}

static void test_disk_persistence(void) {
    printf("[Test 6] Hyperspace Disk Persistence (Save & Load)...\n");
    Node *root1 = NULL;
    for (int i = 0; i < 25; i++) {
        autosophy_autocode_category(&root1, i, i * 3, i * 100, i % 7, (double)i * 0.1, i % 7);
    }

    TreeStats stats1;
    get_tree_stats(root1, &stats1);

    const char *tmp_file = "/tmp/test_hyperspace.dat";
    int save_ok = autosophy_save_tree(root1, tmp_file);
    assert(save_ok == 1);

    Node *root2 = NULL;
    int load_ok = autosophy_load_tree(&root2, tmp_file);
    assert(load_ok == 1);

    TreeStats stats2;
    get_tree_stats(root2, &stats2);

    assert(stats1.node_count == stats2.node_count);
    assert(stats1.max_depth == stats2.max_depth);

    // Verify record contents
    int db_code, gen_id;
    double seed;
    for (int i = 0; i < 25; i++) {
        int found = autosophy_correlate_deep_belief(root2, i, i * 3, &db_code, &gen_id, &seed);
        assert(found == 1);
        assert(db_code == i * 100 && gen_id == i % 7);
    }

    remove(tmp_file);
    free_tree(root1);
    free_tree(root2);
    printf(" -> PASS: Hyperspace disk persistence round-trip verified.\n");
}

static void test_noise_and_transcendence(void) {
    printf("[Test 7] Noise Detection & Transcendental Continued Fraction Approximation...\n");
    Node *root = NULL;

    autosophy_learn(&root, 10, 10, 1.0, 1);
    autosophy_learn(&root, 10, 10, 1.0, 1); // access_count = 2

    // Threshold 2: hits >= 2 is valid knowledge (0 = not noise)
    int is_noise = kernel_detect_noise(root, 10, 10, 2);
    assert(is_noise == 0);

    // Unindexed address: structural noise
    is_noise = kernel_detect_noise(root, 999, 999, 2);
    assert(is_noise == 1);

    // Pi continued fraction convergence test
    double pi_est = kernel_approximate_transcendental(5);
    assert(fabs(pi_est - M_PI) < 1e-8);
    printf(" -> PASS: Noise detection thresholding and Pi convergence verified.\n");

    free_tree(root);
}

int main(void) {
    printf("====================================================\n");
    printf("   AUTOSOPHER COMPREHENSIVE C TEST SUITE            \n");
    printf("====================================================\n");

    test_avl_balancing();
    test_recall_and_nearest();
    test_range_query();
    test_epistemic_updates();
    test_deep_belief_and_autocoding();
    test_disk_persistence();
    test_noise_and_transcendence();

    printf("====================================================\n");
    printf(" ALL 7 C TESTS PASSED CLEANLY!                      \n");
    printf("====================================================\n");
    return 0;
}
