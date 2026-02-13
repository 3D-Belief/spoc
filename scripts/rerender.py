import os
import sys
import argparse
import subprocess
import concurrent.futures
from pathlib import Path

def run_rerender_process(houses_file, output_dir, outdir_prefix, base_dir):
    """
    Run the reconstruct_scene.py script for a single house group file
    
    Parameters:
    houses_file (str): Path to the text file containing house directories
    output_dir (str): Directory to save the output
    outdir_prefix (str): Prefix for output directory names
    base_dir (str): Base directory containing house folders
    
    Returns:
    int: Return code from the process
    """
    # Get the reconstruct_scene.py script path
    script_path = str(Path(__file__).parent.parent / "rerender" / "reconstruct_scene.py")
    
    # Build the command
    cmd = [
        sys.executable,  # Current Python interpreter
        script_path,
        "--process_all",
        "--houses_file", houses_file,
        "--output_dir", output_dir,
        "--outdir_prefix", outdir_prefix,
        "--base_dir", base_dir
    ]
    
    # Extract group info from filename for logging
    group_name = Path(houses_file).stem
    
    print(f"Starting rendering process for {group_name}")
    
    # Run the command
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Create log files
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        
        stdout_file = log_dir / f"{group_name}_stdout.log"
        stderr_file = log_dir / f"{group_name}_stderr.log"
        
        with open(stdout_file, 'w') as stdout_log, open(stderr_file, 'w') as stderr_log:
            # Process output in real-time
            for line in process.stdout:
                print(f"[{group_name}] {line.strip()}")
                stdout_log.write(line)
                stdout_log.flush()
            
            # Get any remaining stderr
            for line in process.stderr:
                print(f"[{group_name} ERROR] {line.strip()}", file=sys.stderr)
                stderr_log.write(line)
                stderr_log.flush()
        
        # Wait for process to finish and get return code
        return_code = process.wait()
        print(f"Process for {group_name} completed with return code {return_code}")
        return return_code
    
    except Exception as e:
        print(f"Error running process for {group_name}: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Run rerendering in parallel on a single GPU')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID (0-7)')
    parser.add_argument('--house_groups_dir', type=str, 
                        default="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/house_groups",
                        help='Directory containing house group files')
    parser.add_argument('--output_base_dir', type=str, 
                        default="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/rerendered",
                        help='Base directory for output')
    parser.add_argument('--base_dir', type=str, 
                        default="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/all/ObjectNavType/val",
                        help='Base directory containing house folders')
    parser.add_argument('--max_workers', type=int, default=20,
                        help='Maximum number of parallel processes')
    parser.add_argument('--prefix_with_group_name', action='store_true',
                        help='Prefix output directories with group name')
    parser.add_argument('--outdir_prefix', type=str, default="",
                        help='Custom prefix for output directories')
    
    args = parser.parse_args()
    
    # Validate GPU ID
    if args.gpu_id < 0 or args.gpu_id > 7:
        print(f"Error: GPU ID must be between 0 and 7. Got {args.gpu_id}")
        return 1
    
    # Set CUDA_VISIBLE_DEVICES to restrict to this GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Find all group files for this GPU
    house_groups_dir = Path(args.house_groups_dir)
    group_files = sorted(house_groups_dir.glob(f"gpu_{args.gpu_id}_group_*.txt"))
    
    if not group_files:
        print(f"Error: No group files found for GPU {args.gpu_id} in {house_groups_dir}")
        return 1
    
    print(f"Found {len(group_files)} group files for GPU {args.gpu_id}")
    
    # Create output directory for this GPU
    gpu_output_dir = Path(args.output_base_dir)
    gpu_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run processes in parallel
    tasks = []
    for group_file in group_files:
        group_name = group_file.stem  # e.g., "gpu_0_group_0"
        
        # Use group name as prefix if specified, otherwise use the custom prefix
        if args.prefix_with_group_name:
            prefix = f"{group_name}_"
        else:
            prefix = args.outdir_prefix
        
        tasks.append((
            str(group_file),
            str(gpu_output_dir),
            prefix,
            args.base_dir
        ))
    
    # Use a smaller number of workers if there are fewer tasks
    actual_workers = min(args.max_workers, len(tasks))
    print(f"Starting {len(tasks)} rendering processes with {actual_workers} workers")
    
    # Run tasks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(run_rerender_process, *task)
            futures.append(future)
        
        # Wait for all tasks to complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                print(f"Task {i+1}/{len(tasks)} completed with result: {result}")
            except Exception as e:
                print(f"Task {i+1}/{len(tasks)} raised an exception: {e}")
    
    print(f"All rendering processes for GPU {args.gpu_id} complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
