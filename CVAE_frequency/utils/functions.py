import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from CVAE.cvae import CVAE
import torch
import plotly.express as px


def load_best_params(json_path="best_params.json"):
    with open(json_path, "r") as f:
        best_params = json.load(f)
    latent_dim = best_params["latent_dim"]
    hidden_dim_list = best_params["hidden_dim_list"]
    lr = best_params["lr"]
    dropout = best_params["dropout"]
    print("Best parameters found:")
    print("Latent dimension:", latent_dim)
    print("Hidden dimensions:", hidden_dim_list)
    print("Learning rate:", lr)
    print("Dropout rate:", dropout)
    return latent_dim, hidden_dim_list, lr, dropout

def pca(dataset, variance_threshold=0.99):
    """
    Calcule le nombre minimal de composantes principales expliquant variance_threshold de la variance totale.
    Affiche la forme concaténée et le nombre de composantes nécessaires.
    """
    X_all = dataset.tensors[0].cpu().numpy()
    Y_all = dataset.tensors[1].cpu().numpy()
    XY_data = np.concatenate([X_all, Y_all], axis=1)
    print("Shape of concatenated data (X, Y):", XY_data.shape)

    pca = PCA()
    pca.fit(XY_data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    latent_dim_min = np.argmax(cumsum >= variance_threshold) + 1

    print(f"Nombre de composantes pour {int(variance_threshold*100)}% de variance : {latent_dim_min}")

def train_cvae(cvae, optimizer, scheduler, num_epochs, dataloader_train, dataloader_test, device):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        cvae.train()

        for x_batch, y_batch, labels_batchs in dataloader_train:
            xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
            labels_batchs = labels_batchs.to(device)
            recon, mu, log_var = cvae(xy_batch, labels_batchs)
            loss_batch = cvae.loss(recon, xy_batch, mu, log_var)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            epoch_loss += loss_batch.item()

        avg_loss = epoch_loss / len(dataloader_train)
        train_losses.append(avg_loss)

        # Validation
        cvae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, labels_val in dataloader_test:
                xy_val = torch.cat([x_val, y_val], dim=1).to(device)
                labels_batchs = labels_val.to(device)
                recon, mu, log_var = cvae(xy_val, labels_batchs)
                loss = cvae.loss(recon, xy_val, mu, log_var)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(dataloader_test)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate (epoch {epoch+1}): {current_lr:.6e}")

    # Affichage des courbes de loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbe de la loss d'entraînement et validation")
    plt.legend()
    plt.show()
 
def save_model(model, optimizer, model_path, input_dim, latent_dim, hidden_dim_list, dropout, scaler_x):
    """
    Sauvegarde le modèle VAE et ses paramètres dans un fichier.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim_list': hidden_dim_list,
        'dropout': dropout,
        'scaler_x': scaler_x
    }
    torch.save(checkpoint, model_path)
    print(f"Modèle sauvegardé sous {model_path}")

def load_model(model_path):
    """
    Charge un modèle VAE sauvegardé et ses paramètres/scalers.
    Retourne : vae, optimizer_state_dict, scaler_x, device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    cvae = CVAE(
        input_dim=checkpoint['input_dim'],
        cond_dim=1,  # Ajoute cette ligne !
        latent_dim=checkpoint['latent_dim'],
        hidden_dim_list=checkpoint['hidden_dim_list'],
        dropout=checkpoint['dropout']
    ).to(device)
    cvae.load_state_dict(checkpoint['model_state_dict'])

    print("Paramètres du modèle chargés :")
    print("Input dimension:", checkpoint['input_dim'])
    print("Latent dimension:", checkpoint['latent_dim'])
    print("Hidden dimensions:", checkpoint['hidden_dim_list'])
    print("Dropout rate:", checkpoint['dropout'])
    print("Modèle chargé avec succès.")

    scaler_x = checkpoint.get('scaler_x', None)
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    return cvae, optimizer_state_dict, scaler_x, device

def display_spectrums(n_examples, original, recon):
    """
    Affiche n_examples spectres originaux et reconstruits côte à côte.
    original : array [N, ...] (ex: orig_y)
    recon    : array [N, ...] (ex: recon_y)
    """
    n_cols = 5
    n_rows = n_examples // n_cols
    plt.figure(figsize=(2.2 * n_cols, 2.5 * n_rows))
    for i in range(n_examples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(original[i], label='Original Y')
        plt.plot(recon[i], label='Recon Y')
        plt.title(f"Ex {i+1}", fontsize=8)
        plt.xticks([])
        plt.yticks([])
        if i % n_cols == 0:
            plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

def cvae_reconstruction(cvae, dataloader, device, dim_x):
    """
    Prend un batch du dataloader, effectue la reconstruction avec le CVAE,
    et affiche les shapes et premiers exemples de X et Y originaux/reconstruits.
    Retourne : orig_x, orig_y, recon_x, recon_y
    """
    cvae.eval()
    with torch.no_grad():
        x_batch, y_batch, labels_batch = next(iter(dataloader))
        xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
        labels_batch = labels_batch.to(device)
        recon, mu, log_var = cvae(xy_batch, labels_batch)
        recon_np = recon.cpu().numpy()
        original_np = xy_batch.cpu().numpy()

    recon_x = recon_np[:, :dim_x]
    recon_y = recon_np[:, dim_x:]
    print("recon_x shape:", recon_x.shape)
    print("recon_y shape:", recon_y.shape)
    print("X normalisé:", recon_x[0])
    print("Y normalisé:", recon_y[0])

    orig_x = original_np[:, :dim_x]
    orig_y = original_np[:, dim_x:]
    print("orig_x shape:", orig_x.shape)
    print("orig_y shape:", orig_y.shape)
    print("X normalisé:", orig_x[0])
    print("Y normalisé:", orig_y[0])

    return orig_x, orig_y, recon_x, recon_y

def cvae_latent_3d(
    cvae, dataloader, index1=3, index2=9, index3=8, color_mode="peak_pos", index4=None, device=None
):
    """
    Affiche une projection 3D de l'espace latent du CVAE pour un batch du dataloader.
    index1, index2, index3 : indices des dimensions latentes à afficher (axes).
    color_mode : "peak_pos" (par défaut), "sum", ou "label" (pour la couleur).
    index4 : indice de la 5e dimension à utiliser pour la taille des points (optionnel).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prendre un batch
    x_batch, y_batch, label_batch = next(iter(dataloader))
    xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
    label_batch = label_batch.to(device)

    with torch.no_grad():
        z_mean, _ = cvae.encode(xy_batch, label_batch)  # [N, latent_dim]

    if color_mode == "peak_pos":
        color = y_batch.argmax(dim=1).cpu().numpy()
        color_label = "Position du pic (indice)"
    elif color_mode == "sum":
        color = y_batch.sum(dim=1).cpu().numpy()
        color_label = "Somme(y)"
    elif color_mode == "label":
        color = label_batch.cpu().numpy().squeeze()
        color_label = "Label"
    else:
        color = None
        color_label = ""

    # Taille des points selon la 5e dimension si index4 est fourni
    if index4 is not None:
        size = z_mean[:, index4].detach().cpu().numpy()
        size = np.abs(size)
        ptp = np.ptp(size)
        if ptp < 1e-6:
            size = np.full_like(size, 20.0)  # taille fixe si pas de variance
        else:
            sqrt_size = np.sqrt(size)
            size = 10 + 30 * (sqrt_size - sqrt_size.min()) / (np.ptp(sqrt_size) + 1e-8)
    else:
        size = None

    fig = px.scatter_3d(
        x=z_mean[:, index1].detach().cpu().numpy(),
        y=z_mean[:, index2].detach().cpu().numpy(),
        z=z_mean[:, index3].detach().cpu().numpy(),
        color=color,
        size=size,
        labels={'color': color_label, 'size': f'z[{index4}]' if index4 is not None else None}
    )
    fig.update_layout(title=f"Latent space: z[{index1}], z[{index2}], z[{index3}]"
                                 + (f", size=z[{index4}]" if index4 is not None else ""))
    fig.show()


def cvae_latent_2d(cvae, dataloader, index_x=3, index_y=9, index_histo=8, device=None):
    """
    Affiche des scatterplots 2D du latent z échantillonné, colorés selon la somme des cibles,
    la position du pic, et la valeur du label.
    Compatible CVAE : attend x, y, label dans le batch.
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_iter = iter(dataloader)
    x_batch, y_batch, label_batch = next(data_iter)
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("label_batch shape:", label_batch.shape)
    
    xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
    label_batch = label_batch.to(device)

    mu, log_var = cvae.encode(xy_batch, label_batch)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + std * eps  # <-- échantillon réel

    # 1. Couleur selon la somme des valeurs cibles
    color_sum = y_batch.sum(dim=1).cpu().numpy()
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        z[:, index_x].detach().cpu().numpy(),
        z[:, index_y].detach().cpu().numpy(),
        c=color_sum, alpha=0.6, cmap='viridis'
    )
    plt.colorbar(sc, label="Somme(y)")
    plt.title("z ~ N(0, I) ? (Somme des valeurs cibles)")
    plt.xlabel(f"z[{index_x}]")
    plt.ylabel(f"z[{index_y}]")
    plt.grid(True)
    plt.show()

    # 2. Couleur selon la position du pic (indice du max)
    color_peak_pos = y_batch.argmax(dim=1).cpu().numpy()
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        z[:, index_x].detach().cpu().numpy(),
        z[:, index_y].detach().cpu().numpy(),
        c=color_peak_pos, alpha=0.6, cmap='viridis'
    )
    plt.colorbar(sc, label="Position du pic (indice)")
    plt.title("z ~ N(0, I) ? (Position du pic)")
    plt.xlabel(f"z[{index_x}]")
    plt.ylabel(f"z[{index_y}]")
    plt.grid(True)
    plt.show()

    # 3. Couleur selon la valeur du label
    color_label = label_batch.cpu().numpy().squeeze()
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        z[:, index_x].detach().cpu().numpy(),
        z[:, index_y].detach().cpu().numpy(),
        c=color_label, alpha=0.6, cmap='tab10' if len(np.unique(color_label)) < 10 else 'viridis'
    )
    plt.colorbar(sc, label="Label")
    plt.title("z ~ N(0, I) ? (Label)")
    plt.xlabel(f"z[{index_x}]")
    plt.ylabel(f"z[{index_y}]")
    plt.grid(True)
    plt.show()

    print("Variance moyenne de z :", z.var(dim=0).mean().item())
    print("Moyenne de z :", z.mean(dim=0).detach().cpu().numpy())
    plt.hist(z[:, index_histo].detach().cpu().numpy(), bins=50)
    plt.title(f"Histogramme de z[{index_histo}]")
    plt.show()

def vae_latent_interpolation(vae, latent_dim=32, dim_x=4, axis=0, z_min=-2, z_max=2, steps=15, fixed_z=None, device=None):
    """
    Affiche l'effet de l'interpolation sur un axe latent donné.
    - vae : modèle VAE
    - dim_x : dimensions de X dans la sortie du décodeur
    - axis : indice de l'axe latent à faire varier
    - z_min, z_max : bornes de variation de l'axe
    - steps : nombre de points à générer
    - fixed_z : vecteur latent de base (sinon zeros)
    - device : cpu ou cuda
    """
    vae.eval()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fixed_z is None:
        z_base = torch.zeros(latent_dim, device=device)
    else:
        z_base = torch.tensor(fixed_z, dtype=torch.float32, device=device)

    z_values = np.linspace(z_min, z_max, steps)
    decoded_y = []

    with torch.no_grad():
        for val in z_values:
            z = z_base.clone()
            z[axis] = val
            out = vae.decode(z.unsqueeze(0)).cpu().numpy()
            y = out[:, dim_x:]  # On suppose que la sortie est [X|Y]
            decoded_y.append(y[0])

    plt.figure(figsize=(10, 2.5 * steps // 5))
    for i, y in enumerate(decoded_y):
        plt.subplot(steps // 5, 5, i + 1)
        plt.plot(y)
        plt.title(f"z[{axis}]={z_values[i]:.2f}", fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.suptitle(f"Interpolation sur z[{axis}] de {z_min} à {z_max}", y=1.02)
    plt.show()

def print_top_latent_std(std_tensor, top_n=3):
    """
    Affiche les valeurs de std du latent dans l'ordre décroissant
    et retourne les indices des top_n plus grandes valeurs.
    """
    std_np = std_tensor.detach().cpu().numpy()
    sorted_indices = std_np.argsort()[::-1]
    print("Std of z_train (sorted):")
    for idx in sorted_indices:
        print(f"z[{idx}] : {std_np[idx]:.4f}")
    top_indices = sorted_indices[:top_n]
    print(f"\nIndices des {top_n} plus grandes std :", top_indices)
    return top_indices

def cvae_latent_umap(
    cvae, dataloader, color_mode="peak_pos", device=None, n_neighbors=15, min_dist=0.1, metric="euclidean"
):
    """
    Affiche une projection 3D UMAP de l'espace latent du CVAE pour un batch du dataloader.
    color_mode : "peak_pos", "sum", ou "label" (pour la couleur).
    """
    import umap
    import plotly.express as px

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prendre un batch
    x_batch, y_batch, label_batch = next(iter(dataloader))
    xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
    label_batch = label_batch.to(device)

    with torch.no_grad():
        z_mean, _ = cvae.encode(xy_batch, label_batch)  # [N, latent_dim]
        z_np = z_mean.cpu().numpy()

    if color_mode == "peak_pos":
        color = y_batch.argmax(dim=1).cpu().numpy()
        color_label = "Position du pic (indice)"
    elif color_mode == "sum":
        color = y_batch.sum(dim=1).cpu().numpy()
        color_label = "Somme(y)"
    elif color_mode == "label":
        color = label_batch.cpu().numpy().squeeze()
        color_label = "Label"
    else:
        color = None
        color_label = ""

    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    z_umap = reducer.fit_transform(z_np)

    fig = px.scatter_3d(
        x=z_umap[:, 0],
        y=z_umap[:, 1],
        z=z_umap[:, 2],
        color=color,
        labels={'color': color_label}
    )
    fig.update_layout(title="Projection UMAP 3D de l'espace latent")
    fig.show()