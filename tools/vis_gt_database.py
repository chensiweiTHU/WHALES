"visualize objects in the ground truth database to verify the correctness of the database"
import open3d as o3d
if __name__ == '__main__':
    print("visualize objects in the ground truth database to verify the correctness of the database")
    # visualize objects in the ground truth database to verify the correctness of the
    pcd_1 = o3d.io.read_point_cloud("data/dolphins/dolphins_gt_database/1_Vehicle_18.bin")
    vis_1 = o3d.visualization.Visualizer()
    vis_1.create_window()
    vis_1.add_geometry(pcd_1)
    vis_1.run()
    