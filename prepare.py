import argparse
import sys
import os
import zipfile
import py7zr

from pathlib import Path
from utils.preparation import calculate_MD5, remove_folder, copy_folder

hashes = {
    "P2ILF_train": "2d84a80ba4aeb8ff8de5c1c59aca3b50",
    "P2ILF_test": "507759d5a10005e4102514a5edc1e9a5",
    "L3D_train": "dca2aab8184331fc8de832c7ce5beec4",
    "L3D_test": "d8ae3262a78fde371b007ac61721b68c",
    "L3D_val": "a67c32245d0edf310b39ccb2f7f504a0"
}

def prepare_P2ILF(in_path, out_path, is_test):
    if is_test:
        # Source: Rahul/StackOverflow. 2010. Unzipping files in Python.
        with zipfile.ZipFile(in_path, 'r') as zip_ref:
            zip_ref.extractall(Path(out_path / "data" / "P2ILF"))

        new_path = Path(out_path / "data" / "P2ILF")
        remove_folder(Path(new_path / "__MACOSX"))
        new_path = Path(new_path / "P2ILF_testData_confidential")
        Path(new_path / "patient4_p2ilf").rmdir()
        for p in ["patient4", "patient11"]:
            remove_folder(Path(new_path / p / "images" / "mha"))
        copy_folder(new_path, Path(out_path / "data" / "P2ILF" / "test"))
        remove_folder(new_path)
    else:
        # Source: Rahul/StackOverflow. 2010. Unzipping files in Python.
        with zipfile.ZipFile(in_path, 'r') as zip_ref:
            zip_ref.extractall(Path(out_path / "data" / "P2ILF"))
        
        new_path = Path(out_path / "data" / "P2ILF")
        remove_folder(Path(new_path / "__MACOSX"))
        Path.mkdir(Path(new_path / "val"))
        new_path = Path(new_path / "P2ILF_MICCAI2022_Edition1")
        copy_folder(new_path, Path(out_path / "data" / "P2ILF" / "train"))
        copy_folder(Path(new_path / "patient1"), Path(out_path / "data" / "P2ILF" / "val" / "patient1"))
        copy_folder(Path(new_path / "patient2"), Path(out_path / "data" / "P2ILF" / "val" / "patient2"))
        remove_folder(new_path)
        remove_folder(Path(out_path / "data" / "P2ILF" / "train" / "patient1"))
        remove_folder(Path(out_path / "data" / "P2ILF" / "train" / "patient2"))

def prepare_L3D(in_path, out_path, subset):
    with py7zr.SevenZipFile(in_path, 'r') as archive:
        archive.extractall(path=Path(out_path / "data" / "L3D"))
        name = subset.split('_')[1].capitalize()
        copy_folder(Path(out_path / "data" / "L3D" / name), Path(out_path / "data" / "L3D" / name.lower()))
        remove_folder(Path(out_path / "data" / "L3D" / name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to dataset .zip file")
    parser.add_argument("--dataset", help="identifier of dataset file")
    args = parser.parse_args()
    if not args.path or not args.dataset:
        print("--path and --dataset are required, please read README for instructions.")
        sys.exit(1)
    if args.dataset not in hashes.keys():
        print("--dataset must be a valid identifier. please see the README for options.")
        sys.exit(1)
    
    location = Path(args.path)
    if not location.is_file():
        print("File not found at given path.")
        sys.exit(1)
    calc_hash = calculate_MD5(location)
    if calc_hash != hashes[args.dataset]:
        print("Unexpected hash for dataset.")
        sys.exit(1)
    
    # Source: Russel Dias/StackOverflow. 2011. Find the current directory and file's directory.
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    Path.mkdir(dir_path / "data", exist_ok=True)

    if args.dataset == "P2ILF_test":
        prepare_P2ILF(location, dir_path, True)
    elif args.dataset == "P2ILF_train":
        prepare_P2ILF(location, dir_path, False)
    elif args.dataset == "L3D_test":
        prepare_L3D(location, dir_path, "test")
    elif args.dataset == "L3D_train":
        prepare_L3D(location, dir_path, "train")
    elif args.dataset == "L3D_val":
        prepare_L3D(location, dir_path, "val")