# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_utils.ipynb.

# %% auto 0
__all__ = ['ModelCheckpoint', 'download_checkpoint']

# %% ../nbs/00_utils.ipynb 4
from tqdm import tqdm

# %% ../nbs/00_utils.ipynb 5
class ModelCheckpoint:
    SMALL = "https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz"
    MEDIUM = "https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-medium.tar.gz"
    LARGE = "https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-large.tar.gz"

# %% ../nbs/00_utils.ipynb 6
def download_checkpoint(checkpoint_url: str, path: str = "../data") -> None:
    """Download a checkpoint from a URL to a local path.

    Args:
        checkpoint_url (str): URL of the checkpoint.
        path (str): Local path to save the checkpoint.
    """
    import requests

    response = requests.get(checkpoint_url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
