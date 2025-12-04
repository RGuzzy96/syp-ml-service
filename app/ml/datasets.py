from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(name: str):
    print(f"Loading dataset: {name}")
    name = name.lower().strip()

    if name in ["wine", "wine quality"]:
        return load_wine_quality_dataset()
    
    if name in ("tumor", "tumor images", "image tumor"):
        return load_tumor_image_dataset()
    
    raise ValueError(f"Dataset '{name}' not supported.")
    
def load_wine_quality_dataset():
    import pandas as pd

    print("Loading Wine Quality dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    print("Dataset loaded with shape:", df.shape)

    y = df["quality"].values
    X = df.drop(columns=["quality"]).values
    print("Features and labels separated.")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Features standardized.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset split into train and test sets.")

    return X_train, y_train, X_test, y_test

def load_tumor_image_dataset():
    # create list of transformations to be performed on data
    tf = transforms.Compose([
        transforms.Resize((128, 128)),              # resize to 128 x 128px
        transforms.ToTensor(),                      # convert to tensor
        transforms.Normalize([0.5]*3, [0.5]*3)      # normalize, each R G or B value becomes -> (input - mean) / std
    ])

    train_dl = DataLoader(
        datasets.ImageFolder("data/Training", tf), # use data from folder and apply transformations
        batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    test_dl = DataLoader(
        datasets.ImageFolder("data/Testing", tf),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # returning data loaders instead of tensors
    return train_dl, test_dl, "classification"