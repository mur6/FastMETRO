import trimesh


def visualize_mesh(*, mesh):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        # mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    # scene.add_geometry(create_point_geom(a_point, "red"))
    scene.show()


def _create_point_geom(point, color):
    geom = trimesh.creation.icosphere(radius=0.0008)
    if color == "red":
        color = [202, 2, 2, 255]
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


def visualize_mesh_and_points(*, mesh, mesh_2=None, red_points=(), blue_points=()):
    scene = trimesh.Scene()
    scene.add_geometry(set_blue(mesh))
    if mesh_2 is not None:
        scene.add_geometry(set_red(mesh))
    for p in red_points:
        scene.add_geometry(_create_point_geom(p, "red"))
    for p in blue_points:
        scene.add_geometry(_create_point_geom(p, "blue"))
    scene.show()


def make_hand_mesh(mano_model, gt_vertices):
    # gt_vertices = torch.transpose(gt_vertices, 2, 1).squeeze(0)
    print(f"gt_vertices: {gt_vertices.shape}")
    mano_faces = mano_model.layer.th_faces
    print(f"mano_faces: {mano_faces.shape}")
    return trimesh.Trimesh(vertices=gt_vertices, faces=mano_faces)


# return trimesh.Trimesh(vertices=gt_vertices.detach().numpy(), faces=mano_faces)
