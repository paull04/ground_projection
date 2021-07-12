import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

intrincs = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

r = o3d.t.io.RealSenseSensor()
r.start_capture()
for x in range(5):
    r.capture_frame()
f = r.capture_frame()
help(f)
rgbd = f.to_legacy_rgbd_image()
pcd = o3d.geometry.PointCloud()
print(np.asanyarray(rgbd.depth))
p = pcd.create_from_rgbd_image(o3d.geometry.RGBDImage().create_from_color_and_depth(rgbd.color, rgbd.depth), intrincs)
o3d.visualization.draw_geometries([p]
                                  )
print(rgbd)
help(rgbd)