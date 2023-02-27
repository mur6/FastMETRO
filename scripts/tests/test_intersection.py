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


def test_dot_product():
    plane_normal = torch.tensor([-0.6810, -0.2870, -0.6737])
    ray_direction = torch.tensor([0.0022, 0.0015, 0.0031])
    print(plane_normal @ ray_direction)

    ray_direction2 = torch.tensor(
        [[0.0022, 0.0015, 0.0031], [-0.0020, -0.0054, 0.0032], [-0.0002, 0.0039, -0.0063]]
    )
    print(ray_direction2.shape)
    print(ray_direction2 @ plane_normal)


def test_matmul_and_argsort():
    # ベクトル v1, v2, ... を作成
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([4, 5, 6])
    v3 = torch.tensor([7, 8, 9])
    v = torch.stack([v1, v2, v3]).float()

    # 別のベクトル w を作成
    w = torch.tensor([-0.5, 0.2, -0.8])

    # 内積の値で並べ替える
    p = v @ w
    # p = torch.matmul(v, w)
    print(p)

    idx = torch.argsort(p)
    sorted_v = v[idx]

    print(sorted_v)


def test_5():
    # test_intersection()
    # test_matmul_and_argsort()
    # N = 4
    points = torch.tensor(
        [
            [-0.1164, -0.3724, -0.7614],
            [1.0364, -0.9977, 0.1118],
            [0.9462, 0.4087, 0.9843],
            [-0.2610, 0.9101, 0.0963],
        ]
    )
    shifted = torch.roll(
        points,
        shifts=1,
        dims=0,
    )
    d = points - shifted
    print(d.shape)
    d = torch.norm(d, dim=1)
    print(d.sum())

    print("#############")
    c = torch.tensor([[1, 1, 0], [1, 1, 1], [0.5, 0, 0]], dtype=torch.float)
    ans = torch.norm(c, dim=1)
    print(ans)


def test_6():
    ring_mesh_vertices = torch.tensor(
        [
            [-0.1164, -0.3724, -0.7614],
            [1.0364, -0.9977, 0.1118],
            [0.9462, 0.4087, 0.9843],
            [-0.2610, 0.9101, 0.0963],
            [-0.1164, -0.3724, -0.7614],
            [1.0364, -0.9977, 0.1118],
            [0.9462, 0.4087, 0.9843],
            [-0.2610, 0.9101, 0.0963],
        ]
    )
    idx = torch.tensor([[0, 1, 2], [0, 2, 4], [2, 1, 3], [1, 5, 4]])
    x = ring_mesh_vertices[idx]
    print(x.shape)


if __name__ == "__main__":
    test_6()
