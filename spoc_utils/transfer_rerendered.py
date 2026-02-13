import os
import shutil
import argparse

def transfer_folders(src_dir, dst_dir, move=False):
    """
    Transfer all folders under src_dir to dst_dir.
    
    Parameters:
        src_dir (str): Path to the source directory.
        dst_dir (str): Path to the destination directory.
        move (bool): If True, move folders instead of copying.
    """
    if not os.path.isdir(src_dir):
        raise ValueError(f"Source directory does not exist: {src_dir}")
    
    os.makedirs(dst_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_path):
            if move:
                print(f"Moving {src_path} -> {dst_path}")
                shutil.move(src_path, dst_path)
            else:
                print(f"Copying {src_path} -> {dst_path}")
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)  # remove if already exists
                shutil.copytree(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer all folders from source to destination.")
    parser.add_argument("src", help="Source directory")
    parser.add_argument("dst", help="Destination directory")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")

    args = parser.parse_args()
    transfer_folders(args.src, args.dst, move=args.move)
