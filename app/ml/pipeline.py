from concurrent.futures import ThreadPoolExecutor
from .datasets import load_dataset
from .shared_training import run_training_loop
from .classical_ops import classical_kernel
from .quantum.quantum_ops import quantum_kernel

def run_experiment_pipeline(config):
    print("Experiment pipeline started with config:", config)
    X_train, y_train, X_test, y_test = load_dataset(config["dataset"])

    with ThreadPoolExecutor() as executor:
        # start loop with classical only
        fut_classical = executor.submit(
            run_training_loop,
            X_train, y_train, X_test, y_test,
            classical_kernel,
            kernel_type="classical"
        )

        # start loop with quantum kernel offloading
        # fut_quantum = executor.submit(
        #     run_training_loop,
        #     X_train, y_train, X_test, y_test,
        #     quantum_kernel,
        #     kernel_type="quantum"
        # )

        # wait for results
        # quantum_result = fut_quantum.result()
        classical_result = fut_classical.result()

    return {
        # "quantum": quantum_result,
        "classical": classical_result
    }