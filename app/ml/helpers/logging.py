from sklearn.metrics import confusion_matrix, classification_report

def make_logger(prefix: str):
    logs = []

    def log(msg: str):
        entry = f"{prefix} {msg}"
        print(entry)        # console
        logs.append(entry)  # accumulate for return
    return log, logs

def log_predictions(prefix, y_true, y_pred):
    logs = []

    def add(msg):
        entry = f"{prefix} {msg}"
        print(entry)
        logs.append(entry)

    add("Predictions vs Actual:")
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        match = "✓" if t == p else "✗"
        add(f"Sample {i:02d}: Pred = {p}, True = {t} {match}")

    add("\nConfusion Matrix:")
    add(str(confusion_matrix(y_true, y_pred)))

    add("\nClassification Report:")
    add(classification_report(y_true, y_pred, digits=4))

    return logs

