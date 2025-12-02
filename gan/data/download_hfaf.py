import os
import subprocess


def get_hfaf_anime(path=os.path.join("data", "HFAF-dataset")):
    """Clone HFAF-dataset under data/ and return the anime folder path."""
    url = "https://github.com/VickkiMars/HFAF-dataset.git"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        subprocess.run(["git", "clone", url, path], check=True)
    return os.path.join(path, "anime")


if __name__ == "__main__":
    data_root = get_hfaf_anime()
