import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

# from timm.scheduler import CosineLRScheduler
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance


def all_loss(pred_output, gt_y, data, faces):
    # print(out_points.shape)
    pcls = Pointclouds(pred_output.view(-1, 20, 3).contiguous())
    verts = data.x.view(-1, 778, 3).contiguous()
    # loss_1, _ = chamfer_distance(pred_output, y)
    meshes = Meshes(verts=verts, faces=faces)
    loss = point_mesh_face_distance(meshes, pcls)
    return F.mse_loss(pred_output, gt_y) + loss


def similarity(x1, x2, **kwargs):
    return 1 - F.leaky_relu(F.cosine_similarity(x1, x2, **kwargs))


def get_loss_3d_plane(verts_3d, pred_normal_v, pred_pca_mean):
    d = (pred_normal_v * pred_pca_mean).sum(dim=-1)  #  a*x_0 + b*y_0 + c*z_0
    pred_normal_v = pred_normal_v.unsqueeze(-2)
    loss_of_plane = (verts_3d * pred_normal_v).sum(dim=(1, 2)) - d
    loss_of_plane = loss_of_plane.pow(2).mean()
    # print(f"loss_of_plane: {loss_of_plane}") # 0.095
    return loss_of_plane


def get_loss_3d_sphere(verts_3d, pred_pca_mean, pred_radius):
    pred_pca_mean = pred_pca_mean.unsqueeze(-2)
    x = (verts_3d - pred_pca_mean).pow(2).sum(dim=-1)  # [32, 20]
    pred_radius = pred_radius.pow(2).unsqueeze(-1)
    # print(f"x: {x.shape}")
    # print(f"pred_radius: {pred_radius.shape}")
    loss_of_sphere = x - pred_radius
    # print(f"loss_of_sphere: {loss_of_sphere.shape}")
    loss_of_sphere = loss_of_sphere.pow(2).mean()
    # print(f"loss_of_sphere: value={loss_of_sphere}") # 0.0004326337
    return loss_of_sphere


def on_circle_loss_wrap(pred_output, data):
    batch_size = pred_output.shape[0]
    verts_3d = data.y.view(batch_size, 20, 3)
    gt_pca_mean = data.pca_mean.view(batch_size, -1).float()
    gt_normal_v = data.normal_v.view(batch_size, -1).float()
    gt_radius = data.radius.float()
    return on_circle_loss_1(pred_output, verts_3d, gt_pca_mean, gt_normal_v, gt_radius)


other_weight = 0.1
weight_of_pca_mean = 100.0
weight_of_normal_v = 1.0 * other_weight
weight_of_radius = 1e4 * other_weight
weight_of_plane = 100.0 * other_weight
weight_of_sphere = 1e5 * other_weight


def get_all_losses():
    weights = (
        weight_of_pca_mean,
        weight_of_normal_v,
        weight_of_radius,
        weight_of_plane,
        weight_of_sphere,
    )
    labels = ("pca_mean", "normal_v", "radius", "plane", "sphere")
    for label, w_value in zip(labels, weights):
        print(f"weights: {label}: {w_value}")


def on_circle_loss(
    pred_pca_mean, pred_normal_v, pred_radius, verts_3d, gt_pca_mean, gt_normal_v, gt_radius
):
    loss_pca_mean = F.mse_loss(pred_pca_mean, gt_pca_mean)
    loss_pca_mean = loss_pca_mean * weight_of_pca_mean
    loss_normal_v = similarity(pred_normal_v, gt_normal_v)
    loss_normal_v = loss_normal_v.pow(2).mean()
    loss_normal_v *= weight_of_normal_v
    loss_radius = F.mse_loss(pred_radius, gt_radius)
    loss_radius *= weight_of_radius
    if False:
        print(f"type: loss_pca_mean: {loss_pca_mean.dtype}")
        print(f"type: loss_normal_v: {loss_normal_v.dtype}")
        print(f"type: loss_radius: {loss_radius.dtype}")
    debug = False
    if debug:
        print(f"gt: pca_mean: {gt_pca_mean.shape}")
        print(f"gt: normal_v: {gt_normal_v.shape}")
        print(f"gt: radius: {gt_radius.shape}")
        print(f"loss: pca_mean: {loss_pca_mean:.07}")  # 0.004
        print(f"loss: normal_v: {loss_normal_v:.07}")  # 0.33
        print(f"loss: radius: {loss_radius:.07}")  # 0.0009

    loss_of_plane = get_loss_3d_plane(verts_3d, pred_normal_v, pred_pca_mean) * weight_of_plane
    loss_of_sphere = get_loss_3d_sphere(verts_3d, pred_pca_mean, pred_radius) * weight_of_sphere
    # print(f"type: loss_of_plane: {loss_of_plane.dtype}")
    # print(f"type: loss_of_sphere: {loss_of_sphere.dtype}")
    if debug:
        print(f"loss: plane: {loss_of_plane:.07}")
        print(f"loss: sphere: {loss_of_sphere:.07}")
        print()
    loss = loss_pca_mean + loss_normal_v + loss_radius + loss_of_plane + loss_of_sphere
    return loss.float()


def on_circle_loss_1(*, pred_output, verts_3d, gt_pca_mean, gt_normal_v, gt_radius):
    # print(f"type: pred_output: {pred_output.dtype}")
    # print(f"type: verts_3d: {verts_3d.dtype}")
    pred_pca_mean = pred_output[:, :3].float()
    pred_normal_v = pred_output[:, 3:6].float()
    pred_radius = pred_output[:, 6:].squeeze(-1).float()
    # print(f"pred_pca_mean: {pred_pca_mean.shape}")
    # print(f"pred_normal_v: {pred_normal_v.shape}")
    # print(f"pred_radius: {pred_radius.shape}")
    return on_circle_loss(
        pred_pca_mean, pred_normal_v, pred_radius, verts_3d, gt_pca_mean, gt_normal_v, gt_radius
    )


if __name__ == "__main__":
    get_all_losses()
