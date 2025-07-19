import hashlib
from pathlib import Path
from shutil import copy

def calculate_MD5(path):
    """
    Calculates MD5 hash of a file at given path, splits file into chunks to prevent loading large files into RAM.
    """
    # Source: user3064538/StackOverflow. 2019. How to calculate the MD5 checksum of a file in Python.
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def remove_folder(path: Path):
    """
    Deletes a folder, including any of its contents.
    """
    if path.is_file():
        path.unlink()
    else:
        for child in path.iterdir():
            remove_folder(child)
        path.rmdir()

def copy_folder(in_path: Path, out_path: Path):
    """
    Copys the content of a given folder, and places it into a new folder.
    """
    out_path.mkdir(exist_ok=True)
    for f in in_path.iterdir():
        if f.is_dir():
            copy_folder(f, out_path / f.relative_to(in_path))
        elif f.is_file() and not (out_path / f.name).is_file():
            copy(f, out_path)