import numpy as np
import torch


def getLinePlaneCollision_np(planeNormal, planePoint, line_vector_1, line_vector_2, epsilon=1e-6):
    rayPoint = line_vector_1
    ray_direction = line_vector_2 - line_vector_1
    n_dot_u = planeNormal.dot(ray_direction)
    # if abs(ndotu) < epsilon:
    #     raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / n_dot_u
    return w + si * ray_direction + planePoint


def getLinePlaneCollision(plane_normal, plane_point, line_vector_1, line_vector_2, epsilon=1e-6):
    rayPoint = line_vector_1
    ray_direction = line_vector_2 - line_vector_1
    n_dot_u = plane_normal @ ray_direction
    # if abs(ndotu) < epsilon:
    #     raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - plane_point
    si = -(plane_normal @ w) / n_dot_u
    return w + si * ray_direction + plane_point


if __name__ == "__main__":
    # # Define plane
    # planeNormal = np.array([0, 0, 1])
    # planePoint = np.array([0, 0, 5])

    # # Define ray
    # line_vector_1 = np.array([0, 0, -10])
    # line_vector_2 = np.array([0, 0, 10])

    # Define plane
    planeNormal = torch.FloatTensor([0, 0, 1])
    planePoint = torch.FloatTensor([0, 0, 0])

    # Define ray
    line_vector_1 = torch.FloatTensor([0, 0, -10])
    line_vector_2 = torch.FloatTensor([0, 0, 10])
    point = getLinePlaneCollision(planeNormal, planePoint, line_vector_1, line_vector_2)
    print(f"intersection at: {point}")
