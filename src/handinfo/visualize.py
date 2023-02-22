import trimesh


def visualize_mesh(*, mesh):
    scene = trimesh.Scene()
    scene.add_geometry(set_blue(mesh))
    # scene.add_geometry(mesh)
    scene.show()


def visualize_points(*, blue_points=(), red_points=(), draw_origin=True):
    scene = trimesh.Scene()
    for p in blue_points:
        scene.add_geometry(_create_point_geom(p, "blue", radius=0.05))
    for p in red_points:
        scene.add_geometry(_create_point_geom(p, "red", radius=0.05))
    if draw_origin:
        # 原点のポイントを描く
        scene.add_geometry(_create_point_geom((0, 0, 0), "green", radius=0.1))
    scene.show()


def _create_point_geom(point, color, *, radius=0.0008):
    geom = trimesh.creation.icosphere(radius=radius)
    if color == "red":
        color = [202, 2, 2, 255]
    elif color == "green":
        color = [2, 212, 2, 255]
    elif color == "yellow":
        color = [220, 220, 2, 255]
    else:
        color = [0, 0, 200, 255]
    geom.visual.face_colors = color
    geom.apply_translation(point)
    return geom


# def visualize_points(*, points):
#     scene = trimesh.Scene()
#     for p in points:
#         scene.add_geometry(_create_point_geom(p, "red"))
#     scene.show()


def set_color(mesh, *, color):
    for facet in mesh.facets:
        # mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    return mesh


def set_blue(mesh):
    blue = [32, 32, 210, 128]
    return set_color(mesh, color=blue)


def set_red(mesh):
    red = [210, 32, 32, 128]
    return set_color(mesh, color=red)


def visualize_mesh_and_points(
    *,
    gt_mesh=None,
    pred_mesh=None,
    red_points=(),
    blue_points=(),
    yellow_points=(),
    draw_origin=False
):
    scene = trimesh.Scene()
    # Blue:教師, Red:予測値
    if gt_mesh is not None:
        scene.add_geometry(set_blue(gt_mesh))
    if pred_mesh is not None:
        scene.add_geometry(set_red(pred_mesh))
    for p in red_points:
        scene.add_geometry(_create_point_geom(p, "red"))
    for p in blue_points:
        scene.add_geometry(_create_point_geom(p, "blue"))
    for p in yellow_points:
        scene.add_geometry(_create_point_geom(p, "yellow"))
    if draw_origin:
        # 原点のポイントを描く
        scene.add_geometry(_create_point_geom((0, 0, 0), "green", radius=0.001))
    scene.show()


def make_hand_mesh(mano_model, gt_vertices):
    # gt_vertices = torch.transpose(gt_vertices, 2, 1).squeeze(0)
    mano_faces = mano_model.layer.th_faces
    # print(f"mano_faces: {mano_faces.shape}")
    return trimesh.Trimesh(vertices=gt_vertices, faces=mano_faces)


def convert_mesh(mesh):
    # a = 115
    # vertices = mesh.vertices[350 : 350 + a, :]
    vertices = mesh.vertices
    faces = mesh.faces[700:]
    # print(f"convert_mesh: vertices: {vertices.shape}")
    # print(f"convert_mesh: faces: {mesh.faces.shape}")
    return trimesh.Trimesh(vertices=vertices, faces=faces)
