import os
import subprocess


def get_hfaf_anime(path=os.path.join("data", "HFAF-dataset")):
    """HFAF-dataset を data 配下に clone して anime フォルダのパスを返す。"""
    url = "https://github.com/VickkiMars/HFAF-dataset.git"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        subprocess.run(["git", "clone", url, path], check=True)
    return os.path.join(path, "anime")


if __name__ == "__main__":
    data_root = get_hfaf_anime()
