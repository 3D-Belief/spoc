import os
import argparse
import shutil

def is_scene_successful(scene_path):
    """
    Check if a scene is successful.
    A successful scene has a 'videos' folder with exactly 3 .mp4 files.
    """
    videos_dir = os.path.join(scene_path, 'videos')
    
    # Check if videos directory exists
    if not os.path.isdir(videos_dir):
        return False
    
    # Count mp4 files
    mp4_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    return len(mp4_files) == 3

def process_root_directory(root_path, remove_unsuccessful=False):
    """
    Process the root directory to count successful scenes.
    Optionally remove unsuccessful scenes.
    """
    if not os.path.isdir(root_path):
        print(f"Error: {root_path} is not a valid directory")
        return
    
    successful_count = 0
    unsuccessful_count = 0
    
    # Get all subdirectories (scenes) in the root directory
    scene_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    for scene_dir in scene_dirs:
        print(f"Processing scene: {scene_dir}")
        scene_path = os.path.join(root_path, scene_dir)
        
        if is_scene_successful(scene_path):
            successful_count += 1
        else:
            unsuccessful_count += 1
            if remove_unsuccessful:
                print(f"Removing unsuccessful scene: {scene_dir}")
                shutil.rmtree(scene_path)
    
    total_scenes = successful_count + unsuccessful_count
    print(f"Total scenes: {total_scenes}")
    print(f"Successful scenes: {successful_count}")
    print(f"Unsuccessful scenes: {unsuccessful_count}")
    
    if remove_unsuccessful:
        print(f"Removed {unsuccessful_count} unsuccessful scenes")

def main():
    parser = argparse.ArgumentParser(description='Count successful scenes in a data directory')
    parser.add_argument('root_dir', type=str, help='Path to the root directory')
    parser.add_argument('--remove', action='store_true', help='Remove unsuccessful scenes')
    
    args = parser.parse_args()
    
    process_root_directory(args.root_dir, args.remove)

if __name__ == '__main__':
    main()
