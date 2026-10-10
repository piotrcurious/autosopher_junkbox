#include "autosopher.h"

// Kernel 1: Noise and Anomaly Detection
int kernel_detect_noise(Node *root, int f1, int f2, int min_frequency_threshold) {
    Node *current = root;
    while (current != NULL) {
        if (current->key_feature1 == f1 && current->key_feature2 == f2) {
            if (current->access_count < min_frequency_threshold) {
                return 1; // Low frequency transient anomaly / noise
            }
            return 0; // Valid structured pattern
        }
        if (f1 < current->key_feature1 || (f1 == current->key_feature1 && f2 < current->key_feature2)) {
            current = current->left;
        } else {
            current = current->right;
        }
    }
    return 1; // Unindexed address tip = Complete structural noise / unknown anomaly
}

// Kernel 2: Transcendental Approximation
double kernel_approximate_transcendental(int address_depth) {
    if (address_depth <= 0) return 3.0;

    // Continued fraction expansion coefficients for Pi
    double crystal_coefficients[] = {7.0, 15.0, 1.0, 292.0, 1.0, 1.0, 1.0, 2.0};
    double denominator = crystal_coefficients[(address_depth - 1) % 8];

    for (int i = address_depth - 2; i >= 0; i--) {
        double term = crystal_coefficients[i % 8];
        denominator = term + (1.0 / denominator);
    }
    return 3.0 + (1.0 / denominator);
}

// Demo function for Noise & Transcendence
void run_noise_transcendence_demo(void) {
    Node *hyperspace_library = NULL;

    printf("\n==================================================\n");
    printf("   DEMO 2: Noise Detection & Transcendence Kernels\n");
    printf("==================================================\n");

    printf("\n--- Initializing Hyperspace Knowledge Crystal ---\n");
    // Standard baseline knowledge patterns
    autosophy_learn(&hyperspace_library, 12, 34, 3.1415, 1);
    autosophy_learn(&hyperspace_library, 56, 78, 2.7182, 1);

    // Reinforce [12, 34] to make it a high-frequency stable pattern
    autosophy_learn(&hyperspace_library, 12, 34, 3.1415, 1);
    autosophy_learn(&hyperspace_library, 12, 34, 3.1415, 1);

    printf("\n--- Running Kernel 1: Noise & Anomaly Detection ---\n");
    int stream_samples[4][2] = {
        {12, 34}, // Established high-frequency pattern
        {56, 78}, // Single access transient pattern
        {99, 42}, // Unseen/Unindexed address tip
        {12, 34}  // High frequency pattern
    };

    for (int i = 0; i < 4; i++) {
        int f1 = stream_samples[i][0];
        int f2 = stream_samples[i][1];
        int is_noise = kernel_detect_noise(hyperspace_library, f1, f2, 2);

        printf("Data Stream Sample [%2d, %2d] => %s\n",
               f1, f2,
               is_noise ? "FLAGGED: NOISE / STRUCTURAL ANOMALY" : "VERIFIED: VALID KNOWLEDGE TIP");
    }

    printf("\n--- Running Kernel 2: Transcendental Number Approximation ---\n");
    printf("Approximating Pi via Address-Chained Continued Fraction Crystal Expansions:\n");
    for (int depth = 1; depth <= 5; depth++) {
        double val = kernel_approximate_transcendental(depth);
        double err = fabs(val - M_PI);
        printf("Hyperspace Address Expansion Depth %d => Approximated Value: %.9f (Err: %.9f)\n",
               depth, val, err);
    }

    free_tree(hyperspace_library);
}
