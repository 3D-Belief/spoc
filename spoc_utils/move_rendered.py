#!/usr/bin/env python3
import os
import shutil
import glob

def clean_scenes_and_count(root_dir, remove_invalid=False):
    """
    Count scene directories that have exactly 3 video files in their videos folder.
    Optionally remove invalid scene directories.
    
    Args:
        root_dir (str): Path to the root directory containing scene directories
        remove_invalid (bool): Whether to remove invalid scene directories
        
    Returns:
        tuple: (valid_scenes_count, invalid_scenes_count)
    """
    valid_scenes = 0
    invalid_scenes = 0
    
    # Get all directories in the root directory
    scene_dirs = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"Found {len(scene_dirs)} scene directories to check...")
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(root_dir, scene_dir)
        videos_path = os.path.join(scene_path, "videos")
        
        # Check if videos directory exists
        if not os.path.exists(videos_path):
            if remove_invalid:
                print(f"Removing {scene_dir}: No videos directory found")
                shutil.rmtree(scene_path)
            else:
                print(f"Invalid {scene_dir}: No videos directory found")
            invalid_scenes += 1
            continue
        
        # Count video files in the videos directory
        video_files = glob.glob(os.path.join(videos_path, "*.mp4"))
        video_count = len(video_files)
        
        if video_count != 3:
            if remove_invalid:
                print(f"Removing {scene_dir}: Found {video_count} video files (expected 3)")
                shutil.rmtree(scene_path)
            else:
                print(f"Invalid {scene_dir}: Found {video_count} video files (expected 3)")
            invalid_scenes += 1
        else:
            print(f"Valid {scene_dir}: Found {video_count} video files ✓")
            valid_scenes += 1
    
    return valid_scenes, invalid_scenes

def main():
    # Set your root directory path here
    root_directory = input("Enter the path to the root directory: ").strip()
    
    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist.")
        return
    
    print(f"Analyzing scenes in: {root_directory}")
    
    # Ask if user wants to remove invalid scenes
    remove_choice = input("Do you want to remove invalid scenes? (y/N): ").strip().lower()
    remove_invalid = remove_choice == 'y'
    
    if remove_invalid:
        print("This will permanently delete scene directories that don't have exactly 3 video files.")
        confirm = input("Are you sure you want to continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
    
    valid_count, invalid_count = clean_scenes_and_count(root_directory, remove_invalid)
    
    print("\n" + "="*50)
    if remove_invalid:
        print(f"Cleanup completed!")
        print(f"Valid scenes remaining: {valid_count}")
        print(f"Invalid scenes removed: {invalid_count}")
    else:
        print(f"Analysis completed!")
        print(f"Valid scenes found: {valid_count}")
        print(f"Invalid scenes found: {invalid_count}")
    print(f"Total scenes processed: {valid_count + invalid_count}")

if __name__ == "__main__":
    main()