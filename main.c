#include "autosopher.h"

void run_hybrid_tf_demo(void) {
    printf("\n==================================================\n");
    printf("   DEMO 4: Hybrid TensorFlow + Autosopher System  \n");
    printf("==================================================\n");
    printf("Invoking python hybrid_tf_autosopher.py benchmark...\n\n");
    int status = system("python3 hybrid_tf_autosopher.py");
    if (status != 0) {
        printf("Error: Hybrid TensorFlow benchmark failed or python3 environment missing.\n");
    }
}

void run_interactive_hyperspace(void) {
    Node *library = NULL;
    int choice;

    printf("\n==================================================\n");
    printf("   DEMO 3: Interactive Autosophy Hyperspace Memory\n");
    printf("==================================================\n");

    while (1) {
        printf("\n--- Interactive Submenu ---\n");
        printf("1. Learn/Inject Pattern [Feature 1, Feature 2] -> Outcome\n");
        printf("2. Recall Pattern Address [Feature 1, Feature 2]\n");
        printf("3. Display Hyperspace Tree & Statistics\n");
        printf("4. Run Anomaly Detection Sweep on Hyperspace\n");
        printf("5. Return to Main Menu\n");
        printf("Enter choice (1-5): ");

        if (scanf("%d", &choice) != 1) {
            while (getchar() != '\n'); // clear buffer
            printf("Invalid input.\n");
            continue;
        }

        if (choice == 1) {
            int f1, f2, outcome;
            double payload;
            printf("Enter Feature 1 (int): ");
            if (scanf("%d", &f1) != 1) continue;
            printf("Enter Feature 2 (int): ");
            if (scanf("%d", &f2) != 1) continue;
            printf("Enter Outcome (int): ");
            if (scanf("%d", &outcome) != 1) continue;
            printf("Enter Data Payload (double): ");
            if (scanf("%lf", &payload) != 1) continue;

            autosophy_learn(&library, f1, f2, payload, outcome);
            printf("SUCCESS: Pattern [%d, %d] self-assembled into Hyperspace Tree!\n", f1, f2);
        } else if (choice == 2) {
            int f1, f2, outcome = -1;
            double payload = 0.0;
            printf("Enter Feature 1 (int): ");
            if (scanf("%d", &f1) != 1) continue;
            printf("Enter Feature 2 (int): ");
            if (scanf("%d", &f2) != 1) continue;

            int hit = autosophy_recall(library, f1, f2, &payload, &outcome);
            if (hit) {
                printf("HIT: Retrieved Pattern [%d, %d] => Outcome: %d | Payload: %.4f\n",
                       f1, f2, outcome, payload);
            } else {
                printf("MISS: Address [%d, %d] not found in Hyperspace Tree.\n", f1, f2);
            }
        } else if (choice == 3) {
            TreeStats stats;
            get_tree_stats(library, &stats);
            printf("\n--- Hyperspace Memory Hierarchy ---\n");
            if (library == NULL) {
                printf("(Tree is empty)\n");
            } else {
                print_tree_structure(library, 0);
            }
            printf("Stats: Node Count: %d | Depth: %d | Memory Usage: %zu bytes\n",
                   stats.node_count, stats.max_depth, stats.memory_bytes);
        } else if (choice == 4) {
            if (library == NULL) {
                printf("Hyperspace tree is empty. Learn some patterns first.\n");
                continue;
            }
            int threshold;
            printf("Enter minimum frequency threshold for valid patterns: ");
            if (scanf("%d", &threshold) != 1) continue;

            int noise_count = 0;
            int total_count = 0;
            printf("\n--- Anomaly Detection Sweep (Threshold: %d) ---\n", threshold);
            sweep_tree_noise(library, threshold, &noise_count, &total_count);
            printf("Sweep Complete: %d / %d nodes flagged as noise/transient.\n", noise_count, total_count);
        } else if (choice == 5) {
            break;
        } else {
            printf("Invalid selection.\n");
        }
    }

    free_tree(library);
}

static void print_banner(void) {
    printf("========================================================\n");
    printf("           AUTOSOPHER DEMO SUITE (C VERSION)           \n");
    printf(" Autonomous Information Processing & Self-Assembly Tree \n");
    printf("========================================================\n");
}

static void print_menu(void) {
    printf("\nMain Menu:\n");
    printf("1. Run Demo 1: Zero-Epoch Forward Learning (Logical XOR)\n");
    printf("2. Run Demo 2: Noise Detection & Transcendence Approximation\n");
    printf("3. Run Demo 3: Interactive Hyperspace Exploration\n");
    printf("4. Run Demo 4: Hybrid TensorFlow + Autosopher Benchmark\n");
    printf("5. Run All Demos Sequentially\n");
    printf("0. Exit Suite\n");
    printf("Select option: ");
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        if (strcmp(argv[1], "--demo1") == 0) {
            run_forward_learning_demo();
            return 0;
        } else if (strcmp(argv[1], "--demo2") == 0) {
            run_noise_transcendence_demo();
            return 0;
        } else if (strcmp(argv[1], "--hybrid") == 0) {
            run_hybrid_tf_demo();
            return 0;
        } else if (strcmp(argv[1], "--all") == 0) {
            run_forward_learning_demo();
            run_noise_transcendence_demo();
            run_hybrid_tf_demo();
            return 0;
        } else {
            printf("Usage: %s [--demo1 | --demo2 | --hybrid | --all]\n", argv[0]);
            return 1;
        }
    }

    print_banner();
    int choice;
    while (1) {
        print_menu();
        if (scanf("%d", &choice) != 1) {
            break;
        }
        switch (choice) {
            case 1:
                run_forward_learning_demo();
                break;
            case 2:
                run_noise_transcendence_demo();
                break;
            case 3:
                run_interactive_hyperspace();
                break;
            case 4:
                run_hybrid_tf_demo();
                break;
            case 5:
                run_forward_learning_demo();
                run_noise_transcendence_demo();
                run_hybrid_tf_demo();
                break;
            case 0:
                printf("Exiting Autosopher Demo Suite. Goodbye!\n");
                return 0;
            default:
                printf("Invalid selection. Please choose 0-5.\n");
                break;
        }
    }
    return 0;
}
