import numpy as np
from src.handinfo.visualize import visualize_points
import torch


def torch_test():
    a = torch.tensor([1, 2, 3])  # .unsqueeze(0)  # torch.arange(3)
    # b = torch.arange(9).view(3, 3)
    b = torch.full((3, 3), 1)
    print(a.shape, b.shape)
    print(a * b)
    print("########################")
    a = torch.tensor([[1, 2, 3], [1, 2, 3]])  # .unsqueeze(0)  # torch.arange(3)
    # b = torch.arange(9).view(3, 3)
    b = torch.full((2, 3, 3), 1).unsqueeze(0)
    print(a.shape)
    a = a.unsqueeze(1)
    print(a.shape)
    print(a * b)
    # a = torch.ones(2, 3).unsqueeze(1)
    # b = torch.ones(2, 3, 3)
    # c = a * b
    # print(c.shape)
    # print("########################")
    # a = torch.ones(1, 3)
    # b = torch.zeros(1, 3)
    # c = torch.cat((a, b), dim=0)
    # print(c.shape)


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


def test_intersection():
    # # Define plane
    # planeNormal = np.array([0, 0, 1])
    # planePoint = np.array([0, 0, 5])

    # # Define ray
    # line_vector_1 = np.array([0, 0, -10])
    # line_vector_2 = np.array([0, 0, 10])

    # Define plane
    plane_normal = torch.FloatTensor([0, 0, 1])
    plane_point = torch.FloatTensor([0, 0, 0])
    plane_point = torch.tensor([-0.6810, -0.2870, -0.6737])

    # Define ray
    line_vector_1 = torch.FloatTensor([0, 0, -10])
    line_vector_2 = torch.FloatTensor([0, 0, 10])
    # lp_1 = torch.tensor(
    #     [
    #         [[0.0003, 0.0128, 0.0055], [-0.0020, 0.0066, 0.0124]],
    #         [[-0.0123, 0.0061, -0.0016], [0.0003, 0.0128, 0.0055]],
    #         [[-0.0020, 0.0066, 0.0124], [-0.0123, 0.0061, -0.0016]],
    #     ]
    # )
    # lp_2 = torch.tensor(
    #     [
    #         [[-0.0020, 0.0066, 0.0124], [-0.0129, -0.0004, 0.0037]],
    #         [[-0.0123, 0.0061, -0.0016], [-0.0020, 0.0066, 0.0124]],
    #         [[-0.0129, -0.0004, 0.0037], [-0.0123, 0.0061, -0.0016]],
    #     ]
    # )
    # line_endpoints = torch.cat((lp_1, lp_2))
    line_endpoints = torch.load("triangle_sides.pt")
    print(f"line_endpoints: {line_endpoints.shape}")

    def _iter_collision_points():
        for line_vector_1, line_vector_2 in line_endpoints:
            p = getLinePlaneCollision(plane_normal, plane_point, line_vector_1, line_vector_2)
            yield p

    points = list(_iter_collision_points())
    print(f"intersection at: {len(points)}")
    visualize_points(points=points[:100])


def test_torch_roll():
    # import torch
    # a = torch.tensor([0, 1, 2, 3, 4, 5])
    # print(a)
    # # tensor([0, 1, 2, 3, 4, 5])

    # a_shift3 = torch.roll(input=a, shifts=3)
    # print(a_shift3)

    a = torch.tensor([0, 1, 2, 3, 4, 5]).view(3, 2)
    a2 = torch.roll(input=a, shifts=1, dims=0)
    a3 = torch.stack((a, a2), dim=1)
    for k in a3:
        print(k)


if __name__ == "__main__":
    test_intersection()
