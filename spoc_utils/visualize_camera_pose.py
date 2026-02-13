import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import argparse
import matplotlib.animation as animation

def load_camera_poses(camera_pose_dir):
    """
    Load all camera pose NPY files from a directory in chronological order.
    """
    pose_files = sorted(glob.glob(os.path.join(camera_pose_dir, "*.npy")))
    
    if not pose_files:
        raise ValueError(f"No camera pose files found in {camera_pose_dir}")
    
    poses = []
    for file_path in pose_files:
        pose = np.load(file_path)
        poses.append(pose)
    
    print(f"Loaded {len(poses)} camera pose files")
    return poses

def extract_camera_trajectory(poses):
    """
    Extract camera positions and orientations from camera pose matrices.
    
    Args:
        poses: List of 4x4 camera-to-world transformation matrices
        
    Returns:
        positions: Array of camera positions [N, 3]
        view_directions: Array of camera view directions [N, 3]
        up_vectors: Array of camera up vectors [N, 3]
    """
    positions = []
    view_directions = []
    up_vectors = []
    
    for pose in poses:
        # Extract position (translation part of the matrix)
        position = pose[:3, 3]
        positions.append(position)
        
        # Extract rotation matrix
        rotation = pose[:3, :3]
        
        # Forward vector is the negative z-axis of the camera coordinate system
        # This is because in most computer vision conventions, camera looks along -z
        forward_vector = -rotation[:, 2]
        view_directions.append(forward_vector)
        
        # Up vector is the y-axis of the camera coordinate system
        up_vector = rotation[:, 1]
        up_vectors.append(up_vector)
    
    return np.array(positions), np.array(view_directions), np.array(up_vectors)

def visualize_camera_trajectory(positions, view_directions, up_vectors, title="Camera Trajectory", 
                               output_file=None, frame_stride=1):
    """
    Visualize camera trajectory with view directions and up vectors.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], 'b-', linewidth=1, alpha=0.5)
    
    # Plot camera positions, view directions, and up vectors
    for i in range(0, len(positions), frame_stride):
        pos = positions[i]
        view_dir = view_directions[i]
        up_vec = up_vectors[i]
        
        # Plot camera position
        ax.scatter(pos[0], pos[2], pos[1], color='red', s=30, alpha=0.7)
        
        # Scale direction vector for visualization
        view_length = 1.0  # 1 meter forward view direction
        up_length = 0.5    # 0.5 meter up vector
        
        # Plot view direction arrow
        ax.quiver(pos[0], pos[2], pos[1], 
                  view_dir[0] * view_length, view_dir[2] * view_length, view_dir[1] * view_length,
                  color='green', arrow_length_ratio=0.2)
        
        # Plot up vector arrow
        ax.quiver(pos[0], pos[2], pos[1], 
                  up_vec[0] * up_length, up_vec[2] * up_length, up_vec[1] * up_length,
                  color='blue', arrow_length_ratio=0.2)
        
        # Add frame number for reference
        ax.text(pos[0], pos[2], pos[1], f"{i}", fontsize=8, color='black')
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_zlabel('Y (meters)')  # Y is up in AI2THOR
    ax.set_title(title)
    
    # Set exact scale with reduced padding to make axes shorter
    x_min, x_max = positions[:, 0].min() - 0.3, positions[:, 0].max() + 0.3
    z_min, z_max = positions[:, 2].min() - 0.3, positions[:, 2].max() + 0.3
    y_min, y_max = positions[:, 1].min() - 0.3, positions[:, 1].max() + 0.3
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(y_min, y_max)
    
    # Create a legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8),
        Line2D([0], [0], color='green', lw=2),
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='b', lw=1, alpha=0.5)
    ]
    ax.legend(custom_lines, ['Camera Position', 'View Direction', 'Up Vector', 'Trajectory Path'], 
             loc='upper right')
    
    # Maintain exact proportions (1:1:1) for x, y, z axes
    ax.set_box_aspect([1, 1, 1])
    
    # Add grid for better spatial reference
    ax.grid(True)
    
    # Reduce the length of axis arrows
    ax.xaxis._axinfo['juggled'] = (0, 0, 0)
    ax.yaxis._axinfo['juggled'] = (1, 1, 1)
    ax.zaxis._axinfo['juggled'] = (2, 2, 2)
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def create_frustum_animation(positions, view_directions, up_vectors, title="Camera Trajectory",
                            output_file="camera_trajectory.mp4"):
    """
    Create an animation showing camera frustums along trajectory.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_zlabel('Y (meters)')
    ax.set_title(title)
    
    # Set exact scale with reduced padding to make axes shorter
    x_min, x_max = positions[:, 0].min() - 0.3, positions[:, 0].max() + 0.3
    z_min, z_max = positions[:, 2].min() - 0.3, positions[:, 2].max() + 0.3
    y_min, y_max = positions[:, 1].min() - 0.3, positions[:, 1].max() + 0.3
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(y_min, y_max)
    
    # Reduce the length of axis arrows
    ax.xaxis._axinfo['juggled'] = (0, 0, 0)
    ax.yaxis._axinfo['juggled'] = (1, 1, 1)
    ax.zaxis._axinfo['juggled'] = (2, 2, 2)
    
    # Create line object for the trajectory
    line, = ax.plot([], [], [], 'b-', linewidth=1, alpha=0.5)
    
    # Create scatter object for camera position
    camera = ax.scatter([], [], [], color='red', s=50)
    
    # Create frustum visualization
    def calculate_frustum_points(position, view_dir, up_vec, fov_h=83.7, fov_v=55.42, near=0.1, far=2.0):
        # Create a right vector using cross product of view and up
        right_vec = np.cross(view_dir, up_vec)
        right_vec = right_vec / np.linalg.norm(right_vec)
        
        # Recalculate up vector to ensure orthogonality
        up_vec = np.cross(right_vec, view_dir)
        up_vec = up_vec / np.linalg.norm(up_vec)
        
        # Convert FOV to radians
        fov_h_rad = np.radians(fov_h)
        fov_v_rad = np.radians(fov_v)
        
        # Calculate frustum dimensions at near and far planes
        near_height = 2 * np.tan(fov_v_rad / 2) * near
        near_width = 2 * np.tan(fov_h_rad / 2) * near
        far_height = 2 * np.tan(fov_v_rad / 2) * far
        far_width = 2 * np.tan(fov_h_rad / 2) * far
        
        # Calculate the 8 corners of the frustum
        near_top_left = position + near * view_dir + (near_height/2) * up_vec - (near_width/2) * right_vec
        near_top_right = position + near * view_dir + (near_height/2) * up_vec + (near_width/2) * right_vec
        near_bottom_right = position + near * view_dir - (near_height/2) * up_vec + (near_width/2) * right_vec
        near_bottom_left = position + near * view_dir - (near_height/2) * up_vec - (near_width/2) * right_vec
        
        far_top_left = position + far * view_dir + (far_height/2) * up_vec - (far_width/2) * right_vec
        far_top_right = position + far * view_dir + (far_height/2) * up_vec + (far_width/2) * right_vec
        far_bottom_right = position + far * view_dir - (far_height/2) * up_vec + (far_width/2) * right_vec
        far_bottom_left = position + far * view_dir - (far_height/2) * up_vec - (far_width/2) * right_vec
        
        return [near_top_left, near_top_right, near_bottom_right, near_bottom_left,
                far_top_left, far_top_right, far_bottom_right, far_bottom_left]
    
    # Initialize frustum lines
    frustum_lines = []
    for _ in range(12):  # 12 edges in a frustum
        line_obj, = ax.plot([], [], [], 'g-', alpha=0.7)
        frustum_lines.append(line_obj)
    
    # Create text object for frame number
    frame_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        camera._offsets3d = (np.array([]), np.array([]), np.array([]))
        for line_obj in frustum_lines:
            line_obj.set_data([], [])
            line_obj.set_3d_properties([])
        frame_text.set_text("")
        return [line, camera] + frustum_lines + [frame_text]
    
    def animate(i):
        # Update trajectory line
        line.set_data(positions[:i+1, 0], positions[:i+1, 2])
        line.set_3d_properties(positions[:i+1, 1])
        
        # Update camera position
        camera._offsets3d = (positions[i:i+1, 0], positions[i:i+1, 2], positions[i:i+1, 1])
        
        # Calculate frustum points with smaller far plane (reduced from 2.0 to 1.0)
        frustum_points = calculate_frustum_points(positions[i], view_directions[i], up_vectors[i], far=1.0)
        
        # Define edges of the frustum (indices of points to connect)
        edges = [
            # Near plane
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Far plane
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Connecting edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Update frustum lines
        for j, (start_idx, end_idx) in enumerate(edges):
            start_point = frustum_points[start_idx]
            end_point = frustum_points[end_idx]
            frustum_lines[j].set_data([start_point[0], end_point[0]], [start_point[2], end_point[2]])
            frustum_lines[j].set_3d_properties([start_point[1], end_point[1]])
        
        # Update frame text
        frame_text.set_text(f"Frame: {i}")
        
        return [line, camera] + frustum_lines + [frame_text]
    
    # Create animation with exact aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add grid for better spatial reference
    ax.grid(True)
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(positions),
                                 init_func=init, blit=False, interval=50)
    
    # Save animation
    ani.save(output_file, writer='ffmpeg', fps=10)
    plt.close()
    print(f"Animation saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize camera trajectory from pose NPY files')
    parser.add_argument('--pose_dir', type=str, required=True,
                       help='Directory containing camera pose NPY files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualizations (optional)')
    parser.add_argument('--frame_stride', type=int, default=5,
                       help='Stride for frame visualization')
    parser.add_argument('--frustum', action='store_true',
                       help='Create animation with camera frustums')
    parser.add_argument('--title', type=str, default="Camera Trajectory",
                       help='Title for the visualization')
    args = parser.parse_args()
    
    # Extract base directory and episode ID from pose_dir if possible
    base_dir = os.path.dirname(args.pose_dir)
    title = args.title
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load camera poses
    camera_poses = load_camera_poses(args.pose_dir)
    
    # Extract camera trajectory
    positions, view_directions, up_vectors = extract_camera_trajectory(camera_poses)
    
    # Make the first frame position (0,0,0)
    first_position = positions[0].copy()
    positions = positions - first_position
    
    # Visualize camera trajectory
    output_file = None
    if args.output_dir:
        output_file = os.path.join(args.output_dir, "camera_trajectory.png")
    
    visualize_camera_trajectory(positions, view_directions, up_vectors, title, output_file, args.frame_stride)
    
    # Create frustum animation if requested
    if args.frustum:
        animation_file = "camera_trajectory_frustum.mp4"
        if args.output_dir:
            animation_file = os.path.join(args.output_dir, "camera_trajectory_frustum.mp4")
        
        create_frustum_animation(positions, view_directions, up_vectors, title, animation_file)

if __name__ == "__main__":
    main()
