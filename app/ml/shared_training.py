import torch
from torch import nn, optim
import time

# ---------------------------------------------------
# NOTE: this currently only supports classification tasks with tabular data to prove out pipeline and approach
# ---------------------------------------------------

def run_training_loop(X_train, y_train, X_test, y_test, kernel_fn, kernel_type):
    prefix = "QUANT:" if kernel_type == "quantum" else "CLASS:"

    print(f"{prefix} Starting training loop with {kernel_type} kernel...")

    # check if GPU is available, fallback to CPU otherwise
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{prefix} Using device: {device}")

    # convert y_train to tensor so we can inspect classes
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    num_classes = len(torch.unique(y_train_tensor))
    print(f"{prefix} Number of classes: {num_classes}")

    print(f"{prefix} Creating model...")
    # create our model with various layers (this will need to be adapted based on dataset as customization expands)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    ).to(device)
    print(f"{prefix} Model created.")

    print(f"{prefix} Setting up optimizer and loss function...")
    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(f"{prefix} Optimizer and loss function set.")
    print(f"{prefix} Starting training...")

    start_time = time.time()

    for epoch in range(10):  # small number of epochs for demonstration
        model.train()
        optimizer.zero_grad()

        print(f"{prefix} Epoch {epoch+1}: Preparing data...")
        # move data to device
        X_batch = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_train, dtype=torch.long).to(device)

        print(f"{prefix} Epoch {epoch+1}: Computing kernel...")
        # compute kernel using custom kernel function passed as fn arg
        K = kernel_fn(X_batch)
        print(f"{prefix} Epoch {epoch+1}: Kernel computed.")

        print(f"{prefix} Epoch {epoch+1}: Performing forward pass...")
        # forward pass
        outputs = model(K)
        print(f"{prefix} Epoch {epoch+1}: Forward pass completed.")

        print(f"{prefix} Epoch {epoch+1}: Computing loss...")
        # compute loss
        loss = loss_fn(outputs, y_batch)
        print(f"{prefix} Epoch {epoch+1}: Loss computed: {loss.item():.4f}")

        print(f"{prefix} Epoch {epoch+1}: Performing backward pass and optimization step...")
        # backward pass and optimization step
        loss.backward()
        optimizer.step()
        print(f"{prefix} Epoch {epoch+1}: Backward pass and optimization step completed.")

        print(f"{prefix} Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        print(f"{prefix} Evaluating model...")
        # create tensors for test data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        K_test = kernel_fn(X_test_tensor)
        test_outputs = model(K_test)
        _, predicted = torch.max(test_outputs, 1)

        accuracy = (predicted == y_test_tensor).float().mean().item()
        print(f"{prefix} Test Accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "training_time": time.time() - start_time
    }
