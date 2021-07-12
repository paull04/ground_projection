import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs


def get_deprojection(intrinsics, ptr, depth_data):
    res = rs.rs2_deproject_pixel_to_point(intrinsics, ptr, depth_data)
    x = res[2]
    y = res[0]
    return y, x, 1


def get_points(intrinsics, points, color_arr, depth_arr):
    return np.array([
        [
            get_deprojection(intrinsics, x, depth_arr[x[1], x[0]]),
            color_arr[x[1], x[0]]
        ] for x in points
    ], dtype=np.float32)


class GroundProject:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.intrinsics = rs.intrinsics
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def start(self):
        self.pipe.start()

    def get_frame(self):
        return self.pipe.wait_for_frames()

    def transform(self, frame, points):
        f = frame
        f = self.align.process(f)
        depth_frame = f.get_depth_frame()
        color_frame = f.get_color_frame()
        color_arr = np.asanyarray(color_frame.get_data())
        depth_arr = np.asanyarray(depth_frame.get_data())

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        new_points = get_points(intrinsics, points, color_arr, depth_arr)
        return new_points

    def __del__(self):
        self.pipe.stop()


if __name__ == "__main__":
    g = GroundProject()
    g.start()
    f = g.get_frame()
    points = np.asarray([[x, y] for x in range(640) for y in range(480)])
    print(points.shape)
    for x in points[:6]:
        print(x[0], x[1])
    new_one = g.transform(f, points)
    x_y = new_one[:, 0, :2]
    rgb = new_one[:, 1]
    print(x_y)
    print(rgb)
    plt.subplot(2,1,1)
    plt.scatter(x_y[:, 0], x_y[:, 1], c=rgb/255.0, s=[1]*len(x_y))
    plt.subplot(2,1,2)
    plt.imshow(np.asanyarray(f.get_color_frame().get_data()))
    plt.show()


