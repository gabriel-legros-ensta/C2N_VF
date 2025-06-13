import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
import random
import os

def standardize(x, y, label):
    """
    Standardise x (features) et laisse y (spectres) normalisé entre 0 et 1.
    """
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)

    scaler_z = StandardScaler()
    label_scaled = scaler_z.fit_transform(label)

    # Pas de standardisation sur y, on le laisse normalisé au pic
    y_scaled = y

    x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
    y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
    label_scaled = torch.tensor(label_scaled, dtype=torch.float32)

    return x_scaled, y_scaled, label_scaled, scaler_x, scaler_z


def load_normalize_data(batch_size=64, shuffle=True, return_dataset=False):
    """
    Charge les données, les standardise, et retourne soit un DataLoader, soit le dataset complet.
    """
    # Chargement des données
    X_data = np.load(os.path.join('data', 'x_data.npy'))
    y_data = np.load(os.path.join('data', 'y_data.npy'))
    labels_data = np.load(os.path.join('data', 'label_data.npy'))
    labels_data = labels_data.reshape(-1, 1)

    # Réduction de Y : on prend 1 point sur 10 (axis=1)
    #y_data = y_data[:, ::10]  # (n, 500)

    # --- Normalisation au pic pour chaque spectre ---
    y_max = np.max(y_data, axis=1, keepdims=True)
    y_max[y_max == 0] = 1  # éviter division par zéro
    y_data = y_data / y_max

    # --- Standardisation ---
    X_scaled, y_scaled, label_scaled, scaler_x, scaler_z = standardize(X_data, y_data, labels_data)

    # Création du dataset avec labels
    dataset = TensorDataset(X_scaled, y_scaled, label_scaled)

    if return_dataset:
        print("Data loaded and normalized. Returning dataset.")
        return dataset, scaler_x, scaler_z

    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print("Data loaded and normalized. Returning DataLoader.")
        return dataloader, scaler_x, scaler_z

def prepare_data_loaders(batch_size=64, test_ratio=0.2, plot_example=True, seed=42):
    """
    Charge, normalise, split, crée les DataLoaders et affiche un exemple.
    Retourne : dataloader_train, dataloader_test, train_dataset, test_dataset, scaler_x, scaler_y, dim_x, dim_y, input_dim, device
    """
    from torch.utils.data import random_split, DataLoader

    # Chargement et normalisation
    dataset, scaler_x, scaler_z = load_normalize_data(batch_size=None, return_dataset=True)

    # Split train/test
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    # Affichage d'un batch
    data_iter = iter(dataloader_train)
    X_batch, y_batch, labels_batch = next(data_iter)
    print("X_batch shape:", X_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("labels_batch shape:", labels_batch.shape)

    if plot_example:
        plt.figure(figsize=(7, 3))
        plt.plot(y_batch[4].cpu().numpy())
        plt.title("Exemple de spectre (y_batch[4])")
        plt.xlabel("Indice")
        plt.ylabel("Valeur")
        plt.show()

    dim_x = X_batch.shape[1]
    dim_y = y_batch.shape[1]
    print("dim_x:", dim_x)
    print("dim_y:", dim_y)

    input_dim = dim_x + dim_y
    print("Input dimension:", input_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    return dataloader_train, dataloader_test, dataset, train_dataset, test_dataset, scaler_x, scaler_z, dim_x, dim_y, input_dim, device

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Important pour reproductibilité sur GPU