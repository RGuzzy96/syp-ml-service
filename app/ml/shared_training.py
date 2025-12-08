import torch
from torch import nn, optim
import time
from sklearn.svm import SVC
from .helpers.logging import log_predictions, make_logger

# ---------------------------------------------------
# NOTE: this currently only supports classification tasks with tabular data to prove out pipeline and approach
# ---------------------------------------------------

def run_svm_with_kernel(X_train, y_train, X_test, y_test, kernel_fn, kernel_type):
    prefix = "QUANT:" if kernel_type == "quantum" else "CLASS:"
    log, logs = make_logger(prefix)

    # start real end-to-end timing so we capture kernel compute + svm train + test eval
    overall_start = time.time()
    kernel_time = 0.0

    log(f"Starting SVM pipeline...")

    # convert the data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)

    # compute training kernel matrix
    log(f"Computing training kernel...")
    t0 = time.time()   # track kernel compute cost
    K_train = kernel_fn(X_train_t)
    kernel_time += time.time() - t0   # accumulate total kernel computation time

    K_train_np = K_train.numpy()

    log(f"Training SVM...")
    start_time = time.time()

    # fit the svm with the precomputed kernel
    svm = SVC(kernel="precomputed")
    svm.fit(K_train_np, y_train)

    training_time = time.time() - start_time
    log(f"SVM training completed in {training_time:.3f}s")

    # compute the test kernel matrix
    log(f"Computing test kernel...")
    t0 = time.time()   # capture test kernel compute cost
    K_test = kernel_fn(X_test_t, X_train_t)
    kernel_time += time.time() - t0   # add to total kernel compute time

    K_test_np = K_test.numpy()

    # run predictions and calculate accuracy
    preds = svm.predict(K_test_np)

    detailed_logs = log_predictions(prefix, y_test, preds)
    logs.extend(detailed_logs)

    accuracy = (preds == y_test).mean()

    log(f"Accuracy: {accuracy:.4f}")

    # overall runtime including kernel compute + svm + test
    total_time = time.time() - overall_start

    return {
        "accuracy": float(accuracy),
        "training_time": training_time,
        "kernel_time": kernel_time,
        "total_time": total_time,
        "logs": logs
    }

def run_nn_training_loop(X_train, y_train, X_test, y_test, kernel_fn, kernel_type):
    prefix = "QUANT:" if kernel_type == "quantum" else "CLASS:"
    log, logs = make_logger(prefix)

    log(f"Starting training loop with {kernel_type} kernel...")

    # check if GPU is available, fallback to CPU otherwise
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    # convert y_train to tensor so we can inspect classes
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    num_classes = len(torch.unique(y_train_tensor))
    log(f"Number of classes: {num_classes}")

    log(f"Creating model...")
    # create our model with various layers (this will need to be adapted based on dataset as customization expands)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    ).to(device)
    log(f"Model created.")

    log(f"Setting up optimizer and loss function...")
    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    log(f"Optimizer and loss function set.")
    log(f"Starting training...")

    start_time = time.time()

    for epoch in range(10):  # small number of epochs for demonstration
        model.train()
        optimizer.zero_grad()

        log(f"Epoch {epoch+1}: Preparing data...")
        # move data to device
        X_batch = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_train, dtype=torch.long).to(device)

        log(f"Epoch {epoch+1}: Computing kernel...")
        # compute kernel using custom kernel function passed as fn arg
        K = kernel_fn(X_batch)
        log(f"Epoch {epoch+1}: Kernel computed.")

        log(f"Epoch {epoch+1}: Performing forward pass...")
        # forward pass
        outputs = model(K)
        log(f"Epoch {epoch+1}: Forward pass completed.")

        log(f"Epoch {epoch+1}: Computing loss...")
        # compute loss
        loss = loss_fn(outputs, y_batch)
        log(f"Epoch {epoch+1}: Loss computed: {loss.item():.4f}")

        log(f"Epoch {epoch+1}: Performing backward pass and optimization step...")
        # backward pass and optimization step
        loss.backward()
        optimizer.step()
        log(f"Epoch {epoch+1}: Backward pass and optimization step completed.")

        log(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        log(f"Evaluating model...")
        # create tensors for test data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        K_test = kernel_fn(X_test_tensor)
        test_outputs = model(K_test)
        _, predicted = torch.max(test_outputs, 1)

        accuracy = (predicted == y_test_tensor).float().mean().item()
        log(f"Test Accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "training_time": time.time() - start_time,
        "logs": logs
    }
