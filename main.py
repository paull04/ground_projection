from abc import *
import numpy as np
import cv2
import pyrealsense2 as rs


pipe = rs.pipeline()
cfg = pipe.start()


class GroundProject(metaclass=ABCMeta):
    def __init__(self, word_points, normal_points):
        self.w_pts = word_points
        self.n_pts = normal_points


class GroundProjectH(GroundProject):
    def __init__(self, world_points, normal_points):
        super(GroundProjectH, self).__init__(world_points, normal_points)
        self.H, _ = cv2.findHomography(self.n_pts, self.w_pts)

    def __call__(self, ptrs):
        return cv2.perspectiveTransform(ptrs, self.H)


class GroundProject3dP(GroundProject):
    def __init__(self, world_points, normal_points, fx, fy, cx, cy, k1, k2, p1, p2):
        super(GroundProject3dP, self).__init__(world_points, normal_points)
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.A = np.array(
            [
                [fx, 0, fy],
                [0, fy, cy],
                [0, 0, 1]
            ]
        )
        self.k1, self.k2, self.p1, self.p2 = k1, k2, p1, p2
        self.distCoeffs = np.array([k1, k2, p1, p2]).reshape([-1, 1])
        self.R, self.T = self.cam_pos()
        self.camMat = np.empty([3, 4])
        for x in range(3):
            for y in range(3):
                self.camMat[x, y] = self.R[x, y]
        for x in range(3):
            self.camMat[3, x] = self.T[0, x]
        self.camMat = self.A.dot(self.camMat)

    def cam_pos(self):
        retval, rvec, tvec = cv2.solvePnP(self.w_pts, self.n_pts, self.A, self.distCoeffs)
        R = cv2.Rodrigues(rvec)
        T = tvec
        return R, T

    def __call__(self, ptrs):
        return [np.dot(self.camMat, x) for x in ptrs]


if __name__ == "__main__":
    f = pipe.wait_for_frames()
    c = np.asanyarray(f.get_color_frame().get_data())
    pixels = np.array([[[453, 412], ], [[220, 403], ], [[297, 304], ], [[407, 296], ]])
    wor = np.array([
        [[4116.0, -879.8751831054688], ], [[4167.0, 702.7423706054688], ],
        [[10781.0, 455.6855773925781], ], [[11915.0, -1647.5037841796875], ]
    ])
    g = GroundProjectH(wor, pixels)
    h = g.H
    p1 = np.array([453, 412, 1]).reshape([-1, 1])
    print(p1)
    p = np.dot(h, p1)
    a = p.reshape(3)
    print(a[:2] / a[2])
    print(g.H)
    print(g(c))
