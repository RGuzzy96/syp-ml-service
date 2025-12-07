from concurrent.futures import ThreadPoolExecutor
from .datasets import load_dataset
from .shared_training import run_svm_with_kernel
from .classical_ops import classical_kernel
from .quantum.quantum_ops import quantum_kernel

# ---------------------------------------------------
#   NOTE: only using SVM here right now, no matter what config says
#   - this is to simplify the initial proof of concept
#   - in the future this should be fully modular and react to what comes in via config
# ---------------------------------------------------

def run_experiment_pipeline(config):
    print("Experiment pipeline started with config:", config)
    X_train, y_train, X_test, y_test = load_dataset(config["dataset"])

    with ThreadPoolExecutor() as executor:
        # start loop with classical only
        fut_classical = executor.submit(
            run_svm_with_kernel,
            X_train, y_train, X_test, y_test,
            classical_kernel,
            kernel_type="classical"
        )

        # start loop with quantum kernel offloading
        fut_quantum = executor.submit(
            run_svm_with_kernel,
            X_train, y_train, X_test, y_test,
            quantum_kernel,
            kernel_type="quantum"
        )

        # wait for results
        quantum_result = fut_quantum.result()
        classical_result = fut_classical.result()

    return {
        "quantum": quantum_result,
        "classical": classical_result
    }