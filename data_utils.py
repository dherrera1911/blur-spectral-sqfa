import os
import torch
import pooch

def load_covariances():
    """
    Ensure 'data/covariances.pt' is present by downloading from OSF if needed,
    then load and return the PyTorch data.
    """
    file_dir = "data"
    file_path = os.path.join(file_dir, "covariances.pt")

    # Direct download link on OSF
    url = "https://osf.io/3zrng/download"

    # If file is missing, download it
    if not os.path.exists(file_path):
        # Ensure the directory exists
        os.makedirs(file_dir, exist_ok=True)

        pooch.retrieve(
            url=url,
            known_hash=None,  # or e.g. 'sha256:<hash>' if you have one
            fname="covariances.pt",  # saved name
            path=file_dir,           # ensure it goes to "data/"
        )

    # Load the file with PyTorch
    return torch.load(file_path)
