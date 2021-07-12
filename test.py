import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

pipe = rs.pipeline()
cfg = pipe.start()

for x in range(5):
  pipe.wait_for_frames()



profile = cfg.get_stream(rs.stream.depth)
intr = profile.as_video_stream_profile().get_intrinsics()
w = []
align_to = rs.stream.color
align = rs.align(align_to)

def mouse_callback(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    print(f"x: {x}, y: {y}")
    res = rs.rs2_deproject_pixel_to_point(intr, [x, y], w[y, x])
    print(res[2], -res[0], -res[1])

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)
pcd = o3d.geometry.PointCloud()

while True:
  pipe.wait_for_frames()
  f = pipe.wait_for_frames()
  f = align.process(f)
  depth = f.get_depth_frame()
  color = f.get_color_frame()
  intr = depth.profile.as_video_stream_profile().intrinsics
  w = np.asanyarray(depth.get_data())
  a = np.asanyarray(color.get_data())
# arr = [rs.rs2_deproject_pixel_to_point(intr, [j, i], w[i, j]) for j in range(640) for i in range(480)]
#  pcd.points = o3d.utility.Vector3dVector(arr)

  cv2.imshow('image', a)
  cv2.resizeWindow('image', 640, 480)

  k = cv2.waitKey(1) & 0xFF
  if k == 27:
    print("ESC 키 눌러짐")
    break

cv2.destroyAllWindows()

pipe.stop()

