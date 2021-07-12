import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import GroundProjectH


def cal_pos(pos, ):
    pos = pos.reshape(3)
    return pos / pos[2]


class GroundProject(GroundProjectH):
    def __init__(self, world_points, normal_points):
        super(GroundProject, self).__init__(world_points, normal_points)
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        return self.cap.read()[1]

    def transform(self, frame, points):
        new_one = [[cal_pos(np.dot(self.H, x)), frame[x[1, 0], x[0, 0]]] for x in points]
        return np.asarray(new_one)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


#if __name__ == '__main__':
while True:
    pixels = np.array([[[453, 412], ], [[220, 403], ], [[297, 304], ], [[407, 296], ]])
    wor = np.array([
        [[4116.0, -879.8751831054688], ], [[4167.0, 702.7423706054688], ],
        [[10781.0, 455.6855773925781], ], [[11915.0, -1647.5037841796875], ]
    ])
    g = GroundProject(wor, pixels)
    f = g.get_frame()
    points = np.asarray([np.asarray([x, y, 1]).reshape(3, 1) for x in range(640) for y in range(480)])
    print(points.shape)

    for x in points[:5]:
        print(f[x[1, 0], x[0, 0]])

    new_one = g.transform(f, points)
    x_y = new_one[:, 0, :2]
    rgb = new_one[:, 1]
    print(x_y)
    print(rgb)
    plt.subplot(2,1,1)
    plt.scatter(x_y[:, 0], x_y[:, 1], c=rgb/255.0, s=[1]*len(x_y))
    plt.subplot(2,1,2)
    plt.imshow(f)
    plt.show()

