from pathlib import Path
import json
import struct
import numpy as np
from PIL import Image
import pytest
import trimesh
import yaml

from fure_defur import geometry as g
from fure_defur.cameras import decompose
from fure_defur.io import annotations, face_labels, export_obj, mesh
from fure_defur.tangents import orient, generate
from fure_defur.package import write_sdf, ray_scene


def test_nonfur_and_calibration():
    parts = g.calibrate_lengths(np.array([.02, .04, .5]), np.array([0, 0, 1]), ["body", "horns"], 10, {"body": 2})
    assert parts["body"]["v1_length_cm"] == pytest.approx(.035*10*3.4)
    assert parts["horns"]["v2_length_cm"] == 0


def test_correspondence_rejects_wrong_normals():
    m = trimesh.creation.icosphere(subdivisions=1)
    v, f = np.asarray(m.vertices), m.faces
    n, e = g.normals_and_edges(v, f)
    shell = dict(verts=v, normals=-n, shell_width=np.ones(len(v))*.02, reliability_mask_init=np.ones(len(v), bool))
    labels = np.zeros(len(v), int)
    assert not g.transfer_widths(v, n, labels, shell, .001)["supported"].any()
    corrected, info = g.align_shell_normal_frame(v, n, shell, .001)
    assert info["normal_sign"] == -1
    assert g.transfer_widths(v, n, labels, corrected, .001)["supported"].all()


def test_anchors_and_bounds():
    m = trimesh.creation.icosphere(subdivisions=1)
    v, f = np.asarray(m.vertices), m.faces
    n, e = g.normals_and_edges(v, f)
    labels = np.zeros(len(v), int)
    bound = np.full(len(v), .03)
    pinned = v[:, 2] > .8
    d, info = g.solve_displacement(bound*2, bound, np.ones(len(v)), bound, pinned, g.laplacian(v, e, labels))
    assert (d >= 0).all() and (d <= bound+1e-8).all()
    assert (d[pinned] == 0).all()


def test_containment_guard_backs_off_outside_vertices():
    from fure_defur.pipeline import containment_guard
    model = trimesh.creation.icosphere(subdivisions=1)
    vertices, faces = np.asarray(model.vertices), np.asarray(model.faces)
    normals, edges = g.normals_and_edges(vertices, faces)
    initial = np.full(len(vertices), .01)
    # Reversing the normals makes the trial displacement expand the sphere.
    # The guard should back it down to the numerical containment tolerance.
    adjusted, report = containment_guard(vertices, faces, -normals, initial, edges)
    assert report["outside_vertices"] == 0
    assert report["history"][0]["outside_vertices"] == len(vertices)
    assert np.max(adjusted) < np.max(initial)


def test_annotation_overlap_is_error(tmp_path):
    p = tmp_path / "parts.json"
    p.write_text(json.dumps(dict(face=[0, 1], body=[1, 2])))
    with pytest.raises(ValueError, match="overlapping"):
        annotations(p, 3)


def test_nonfur_face_guard():
    f = np.array([[0, 1, 2]])
    assert face_labels(f, np.array([0, 0, 1]), np.array([False, False, True])).tolist() == [1]


def test_obj_export_roundtrips_float64_and_topology(tmp_path):
    model = trimesh.creation.icosphere(subdivisions=2)
    vertices = np.asarray(model.vertices) + np.random.default_rng(11).normal(scale=1e-10, size=model.vertices.shape)
    path = tmp_path / "precise.obj"
    export_obj(path, vertices, model.faces)
    loaded = mesh(path)
    assert np.array_equal(loaded.vertices, vertices)
    assert np.array_equal(loaded.faces, model.faces)


def test_projection_roundtrip():
    k = np.array([[900, 0, 320], [0, 850, 240], [0, 0, 1.]])
    pose = np.eye(4)
    pose[:3, 3] = [.4, -.3, 4.]
    recovered_k, recovered_pose = decompose(k @ pose[:3])
    assert np.allclose(k, recovered_k)
    assert np.allclose(pose, recovered_pose)


def test_prepare_and_colmap_camera_roundtrip(tmp_path):
    from fure_defur.prepare import prepare, import_colmap
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    projections = []
    for i in range(3):
        Image.new("RGB", (32, 24), (64, 128, 190)).save(images / f"v{i}.png")
        Image.new("L", (32, 24), 255).save(masks / f"v{i}.png")
        extr = np.eye(4)
        extr[:3, 3] = [.1*i, 0, 6]
        k = np.array([[40.+2*i, 0, 16+i], [0, 40.+i, 12-.5*i], [0, 0, 1]])
        p = np.eye(4)
        p[:3] = k @ extr[:3]
        projections.append(p)
    path = tmp_path / "p.npy"
    np.save(path, projections)
    points = tmp_path / "points.ply"
    trimesh.PointCloud(np.random.default_rng(2).normal(size=(120, 3))*.1).export(points)
    prepared = tmp_path / "prepared"
    prepare(images, masks, path, prepared, [-1, -1, -1, 1, 1, 1], points)
    converted = tmp_path / "converted.npy"
    import_colmap(prepared / "frosting_scene/sparse/0", prepared / "frosting_scene/images", converted)
    got, expected = np.load(converted)[:, :3], np.load(prepared / "views/projection.npy")[:, :3]
    # Projection matrices are homogeneous: the scene normalization changes their scale.
    expected /= np.linalg.norm(expected[:, 2, :3], axis=1)[:, None, None]
    assert np.allclose(got, expected)
    ks = [decompose(value)[0] for value in np.load(converted)]
    assert all(np.allclose(k, ks[0]) for k in ks)
    size = Image.open(prepared / "neus2/images/0000.png").size
    assert np.allclose(ks[0][:2, 2], np.array(size)/2)
    assert Image.open(prepared / "neus2/images/0000.png").mode == "RGBA"


def test_binary_colmap_camera_import(tmp_path):
    from fure_defur.prepare import import_colmap

    model = tmp_path / "model"
    images = tmp_path / "images"
    model.mkdir()
    images.mkdir()
    Image.new("RGB", (32, 24)).save(images / "view.png")

    (model / "cameras.bin").write_bytes(
        struct.pack("<QiiQQdddd", 1, 7, 1, 32, 24, 40, 42, 16, 12)
    )
    (model / "images.bin").write_bytes(
        struct.pack("<Qidddddddi", 1, 3, 1, 0, 0, 0, .1, -.2, 5, 7)
        + b"view.png\0"
        + struct.pack("<Q", 0)
    )

    output = tmp_path / "projection.npy"
    import_colmap(model, images, output)
    expected = np.array([[40, 0, 16, 84], [0, 42, 12, 51.6], [0, 0, 1, 5]])
    assert np.allclose(np.load(output)[0, :3], expected)


def test_prepare_skew_check_is_scale_aware():
    from fure_defur.prepare import _check_skew
    # Float32 projection decomposition can introduce a few thousandths of a
    # pixel of skew at long focal lengths; that is numerical noise, not a
    # physically skewed camera.
    _check_skew(np.array([[5400., .003, 960.], [0., 5400., 540.], [0., 0., 1.]]))
    with pytest.raises(ValueError, match="camera skew"):
        _check_skew(np.array([[5400., .03, 960.], [0., 5400., 540.], [0., 0., 1.]]))


def test_sign_pass_all_components():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    pair = trimesh.util.concatenate([mesh, mesh.copy().apply_translation([4, 0, 0])])
    t = np.random.default_rng(5).normal(size=(len(pair.faces), 3))
    oriented, count = orient(pair.faces, t)
    assert count > 0
    second, remaining = orient(pair.faces, oriented)
    assert remaining == 0
    assert np.array_equal(oriented, second)


def test_sdf_sign_and_xyz_order(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale([.8, .5, .3])
    write_sdf(tmp_path, mesh.vertices, mesh.faces, 24)
    values = np.load(tmp_path / "sdf_grid.npy")
    assert values[12, 12, 12] < 0
    assert values[0, 0, 0] > 0
    assert values.shape == (24, 24, 24)


def test_real_directional_no_fallback(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "vendor/Directional/include").exists():
        pytest.skip("Directional optional engine not fetched")
    model = trimesh.creation.icosphere(subdivisions=2)
    mesh = tmp_path / "furless.obj"
    model.export(mesh)
    report = generate(mesh, tmp_path, repo)
    assert report["residual_sign_flips"] == 0
    assert report["abs_normal_dot_max"] < 1e-4


def test_training_command_contract(tmp_path):
    from fure_defur.reconstruction import commands
    jobs = commands(tmp_path / "input", tmp_path / "output", 2, "python", "python", 60000)
    assert [j["name"] for j in jobs] == ["3dgs", "neus2", "frosting"]
    assert "--disable_viewer" not in jobs[0]["command"]
    assert jobs[0]["command"][jobs[0]["command"].index("-m")+1].endswith("gs_checkpoint")
    assert all(j["environment"] == {"CUDA_VISIBLE_DEVICES": "2"} for j in jobs)


def test_clean_surface_prefers_closed_subject_over_open_sheet(tmp_path):
    from fure_defur.reconstruction import clean_surface
    subject = trimesh.creation.icosphere(subdivisions=2, radius=1.)
    subject.apply_translation([0, 0, 2])
    sheet = trimesh.creation.box(extents=[10., .02, 10.])
    sheet.update_faces(np.arange(len(sheet.faces)-1))
    source = tmp_path / "raw.obj"
    trimesh.util.concatenate([sheet, subject]).export(source)
    report = clean_surface(source, tmp_path / "clean.obj", iterations=0)
    cleaned = trimesh.load(tmp_path / "clean.obj", process=False, force="mesh")
    assert report["component_selection"] == "largest_significant_watertight"
    assert cleaned.is_watertight
    assert len(cleaned.faces) == len(subject.faces)
    assert report["component_count"] == 1
    assert report["boundary_edges"] == 0
    assert report["nonmanifold_edges"] == 0
    assert report["self_intersection_pairs"] == 0


def test_clean_surface_refuses_large_hole_without_leaving_output(tmp_path):
    from fure_defur.reconstruction import clean_surface
    model = trimesh.creation.icosphere(subdivisions=2)
    model.update_faces(np.arange(len(model.faces) - 1))
    source = tmp_path / "open.obj"
    output = tmp_path / "clean.obj"
    model.export(source)
    with pytest.raises(ValueError, match="unsafe holes"):
        clean_surface(source, output, iterations=0)
    assert not output.exists()


def test_clean_surface_explicit_vertex_rounding_closes_near_duplicate_seam(tmp_path):
    from fure_defur.io import export_obj
    from fure_defur.reconstruction import clean_surface
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [1e-7, 0., 0.]])
    faces = np.array([[0, 2, 1], [4, 1, 3], [0, 3, 2], [1, 2, 3]])
    source = tmp_path / "near_duplicate.obj"
    export_obj(source, vertices, faces)
    report = clean_surface(source, tmp_path / "clean.obj", iterations=0, merge_vertex_digits=5)
    assert report["merge_vertex_digits"] == 5
    assert report["component_count"] == 1
    assert report["watertight"]


def test_clean_surface_rejects_unsafe_vertex_rounding_setting(tmp_path):
    from fure_defur.reconstruction import clean_surface
    source = tmp_path / "mesh.obj"
    trimesh.creation.icosphere(subdivisions=1).export(source)
    with pytest.raises(ValueError, match="0 to 15"):
        clean_surface(source, tmp_path / "clean.obj", merge_vertex_digits=16)
    with pytest.raises(ValueError, match="nonnegative integer"):
        clean_surface(source, tmp_path / "negative.obj", iterations=-1)


def test_partitioned_collision_matches_full_open3d():
    import open3d as o3d
    from fure_defur.pipeline import intersections
    first = trimesh.creation.icosphere(subdivisions=2)
    second = first.copy().apply_translation([.65, .1, .2])
    model = trimesh.util.concatenate([first, second])
    brute = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(model.vertices),
                                     o3d.utility.Vector3iVector(model.faces))
    expected = {tuple(sorted(p)) for p in np.asarray(brute.get_self_intersecting_triangles()).tolist()}
    assert expected
    assert intersections(model.vertices, model.faces) == expected


def test_neus2_staging_never_modifies_original(tmp_path):
    from fure_defur.neus2_input import stage_scene
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGBA", (8, 8), (1, 2, 3, 128)).save(source / "view.png")
    value = dict(frames=[dict(file_path="view.png")])
    scene = source / "transform.json"
    scene.write_text(json.dumps(value))
    (source / "transforms.json").write_text(json.dumps(value))
    staged = stage_scene(scene, tmp_path / "run/input")
    assert list(staged.glob("*.json")) == [staged / "transform.json"]
    assert (staged / "images/000000.png").read_bytes() == (source / "view.png").read_bytes()
    assert json.loads(scene.read_text()) == value
    assert not (source / "output").exists()


def _write_minimal_strand_package(root: Path) -> None:
    from fure_defur.io import digest, export_obj, write_json
    model = trimesh.creation.icosphere(subdivisions=1)
    vertices, faces = np.asarray(model.vertices), np.asarray(model.faces)
    export_obj(root / "neus.obj", vertices, faces)
    export_obj(root / "furless.obj", vertices, faces)

    normals = np.asarray(model.face_normals)
    reference = np.tile([1., 0., 0.], (len(faces), 1))
    reference[np.abs(normals[:, 0]) > .9] = [0., 1., 0.]
    tangent = np.cross(normals, reference)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent, _ = orient(faces, tangent)
    np.save(root / "tan_furless.npy", tangent)
    np.save(root / "field_furless_reshaped.npy", tangent)
    raw = np.column_stack([tangent, tangent])
    with (root / "rawFaceField.dmat").open("w") as handle:
        handle.write(f"{len(faces)}\n6\n")
        handle.write(" ".join(f"{value:.17g}" for value in raw.T.ravel()))
        handle.write("\n")

    vertex_ids = list(range(len(vertices)))
    face_ids = list(range(len(faces)))
    write_json(root / "annotations_furless_reshaped2.json", {"body": vertex_ids})
    write_json(root / "annotations_faces.json", {"body": face_ids})
    count = len(vertices)
    np.savez_compressed(root / "defurring_fields.npz", displacement=np.zeros(count),
                        labels=np.zeros(count, dtype=np.int32), confidence=np.ones(count),
                        transferred_width=np.full(count, .02), nearest_width=np.full(count, .02),
                        upper=np.full(count, .05))

    for folder in ("images_2", "masks_2/body", "masks_2/hair", "masks_2/bald",
                   "masks_2/foreground", "part_masks", "orientations_2/angles",
                   "orientations_2/vars"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (80, 100, 120)).save(root / "images_2/0000.png")
    for name, value in (("body", 255), ("hair", 255), ("bald", 0), ("foreground", 255)):
        Image.new("L", (8, 8), value).save(root / "masks_2" / name / "0000.png")
    Image.fromarray(np.ones((8, 8), dtype=np.uint16)).save(root / "part_masks/0000.png")
    write_json(root / "part_masks/legend.json", {"0": "background", "1": "body"})
    Image.new("L", (8, 8), 0).save(root / "orientations_2/angles/0000.png")
    np.save(root / "orientations_2/vars/0000.npy", np.zeros((8, 8), dtype=np.float32))

    k = np.array([[12., 0, 4], [0, 12., 4], [0, 0, 1.]])
    extr = np.eye(4)
    extr[2, 3] = 3.
    projection = np.eye(4)
    projection[:3] = k @ extr[:3]
    np.save(root / "projection.npy", projection[None].astype(np.float32))

    write_sdf(root, vertices, faces, 8)
    metadata = dict(schema_version=1, data_root=".", variant="shape",
                    mesh="furless.obj", outer_mesh="neus.obj",
                    tangent="tan_furless.npy", vertex_parts="annotations_furless_reshaped2.json",
                    face_parts="annotations_faces.json", projection="projection.npy", images="images_2",
                    masks="masks_2", orientations="orientations_2", sdf="sdf_grid.npy",
                    nonfur_parts=[],
                    cm_per_scene_unit=10., scale_basis="synthetic exact scale",
                    length_cm={"body": 2.}, length_scene={"body": .2},
                    width_cm={"body": .1}, thickness_cm={"body": .1},
                    topology_sha256=digest(root / "furless.obj"))
    write_json(root / "strand_package.json", metadata)
    from fure_defur.package import identity_scale_bytes
    (root / "scale.pickle").write_bytes(identity_scale_bytes())
    adapter = dict(sdf_penetration=dict(sdf_path="sdf_grid.npy", bounds_min="min_bound.npy",
                                        bounds_max="max_bound.npy"),
                   sdf_chamfer=dict(path_to_mesh="neus.obj", path_to_scale=""),
                   textured_strands=dict(path_to_mesh="furless.obj",
                                         path_to_classes="annotations_furless_reshaped2.json",
                                         precomputed_tangent="tan_furless.npy",
                                         path_to_scale="",
                                         mapping_length={"body": .2},
                                         use_metrical_space=False,
                                         normalize_decoder_output=False))
    (root / "hair_config.yaml").write_text(yaml.safe_dump(adapter))


def test_validate_checks_outer_topology_and_training_contract(tmp_path):
    from fure_defur.validation import validate
    _write_minimal_strand_package(tmp_path)
    report = validate(tmp_path)
    assert report["components"] == 1
    assert report["outer_components"] == 1
    assert report["self_intersection_pairs"] == 0
    assert report["outer_self_intersection_pairs"] == 0

    (tmp_path / "scale.pickle").unlink()
    with pytest.raises(ValueError, match="scale.pickle"):
        validate(tmp_path)


def test_validate_rejects_metadata_escape_and_incomplete_face_map(tmp_path):
    from fure_defur.io import write_json
    from fure_defur.validation import validate
    _write_minimal_strand_package(tmp_path)
    metadata_path = tmp_path / "strand_package.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tangent"] = "../tan_furless.npy"
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="escapes the package"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    annotations_path = tmp_path / "annotations_faces.json"
    annotations = json.loads(annotations_path.read_text())
    annotations["body"].pop()
    write_json(annotations_path, annotations)
    with pytest.raises(ValueError, match="leave 1 elements unassigned"):
        validate(tmp_path)


def test_validate_rejects_bad_legend_fields_and_sdf_bounds(tmp_path):
    from fure_defur.io import write_json
    from fure_defur.validation import validate
    _write_minimal_strand_package(tmp_path)
    write_json(tmp_path / "part_masks/legend.json", {"0": "background", "1": "wrong"})
    with pytest.raises(ValueError, match="legend differs"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    with np.load(tmp_path / "defurring_fields.npz") as fields:
        values = {key: fields[key] for key in fields.files}
    values["confidence"] = values["confidence"][:-1]
    np.savez_compressed(tmp_path / "defurring_fields.npz", **values)
    with pytest.raises(ValueError, match="confidence must have one finite value per vertex"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    np.save(tmp_path / "max_bound.npy", np.full(3, .5, dtype=np.float32))
    with pytest.raises(ValueError, match="do not contain the root mesh"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    sdf = np.load(tmp_path / "sdf_grid.npy")
    sdf.flat[0] += 1
    np.save(tmp_path / "sdf_grid.npy", sdf)
    with pytest.raises(ValueError, match="SDF samples disagree"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    np.save(tmp_path / "orientations_2/vars/0000.npy", -np.ones((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="Orientation variance"):
        validate(tmp_path)


def test_validate_rejects_wrong_directional_field_shape(tmp_path):
    from fure_defur.validation import validate
    _write_minimal_strand_package(tmp_path)
    np.save(tmp_path / "field_furless_reshaped.npy", np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="one finite xyz vector per face"):
        validate(tmp_path)


def test_validate_rejects_inconsistent_root_nonfur_and_masks(tmp_path):
    from fure_defur.io import write_json
    from fure_defur.validation import validate

    _write_minimal_strand_package(tmp_path)
    with np.load(tmp_path / "defurring_fields.npz") as fields:
        values = {key: fields[key] for key in fields.files}
    values["displacement"][0] = .01
    np.savez_compressed(tmp_path / "defurring_fields.npz", **values)
    with pytest.raises(ValueError, match="recorded normal displacement"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    metadata_path = tmp_path / "strand_package.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["nonfur_parts"] = ["body"]
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="must have zero length_cm"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    Image.new("L", (8, 8), 0).save(tmp_path / "masks_2/hair/0000.png")
    with pytest.raises(ValueError, match="Hair mask disagrees"):
        validate(tmp_path)


def test_part_tables_require_default_nonfur_names():
    from fure_defur.validation import _part_tables
    metadata = dict(cm_per_scene_unit=10., nonfur_parts=[],
                    length_cm={"eyes": 0.}, length_scene={"eyes": 0.},
                    width_cm={"eyes": 0.}, thickness_cm={"eyes": 0.})
    with pytest.raises(ValueError, match="omits default non-fur"):
        _part_tables(metadata, ["eyes"])


def test_eye_distance_scale_check(tmp_path):
    from fure_defur.pipeline import validate_scale
    eyes = tmp_path / "eyes.ply"
    trimesh.points.PointCloud([[0., 0., 0.], [.2, 0., 0.]]).export(eyes)
    config = dict(cm_per_scene_unit=25., eye_distance_cm=5.)
    report = validate_scale(config, eyes)
    assert report["method"] == "inter_eye_distance"
    assert report["landmark_distance_scene"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="disagrees"):
        validate_scale(dict(cm_per_scene_unit=10., eye_distance_cm=5.), eyes)
    with pytest.raises(ValueError, match="must be set together"):
        validate_scale(dict(cm_per_scene_unit=25.), eyes)
    with pytest.raises(ValueError, match="must be set together"):
        validate_scale(dict(cm_per_scene_unit=25., eye_distance_cm=5.), None)
    with pytest.raises(ValueError, match="cm_per_scene_unit"):
        validate_scale(dict(cm_per_scene_unit=True), None)


def test_orientation_generation_is_worker_deterministic(tmp_path):
    from fure_defur.io import digest
    from fure_defur.orientations import generate

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    grid = np.indices((16, 16)).sum(0).astype(np.uint8) * 8
    for index in range(2):
        rgb = np.stack([grid, np.roll(grid, index + 1, axis=1), 255 - grid], axis=-1)
        Image.fromarray(rgb).save(images / f"{index:04d}.png")
        Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(masks / f"{index:04d}.png")

    sequential = tmp_path / "sequential"
    threaded = tmp_path / "threaded"
    generate(images, masks, sequential, workers=1)
    generate(images, masks, threaded, workers=2)
    for relative in ("angles/0000.png", "angles/0001.png", "vars/0000.npy",
                     "vars/0001.npy", "vis_imgs/0000.png", "vis_imgs/0001.png",
                     "metadata.json"):
        assert digest(sequential / relative) == digest(threaded / relative)


def test_validate_rejects_adapter_eye_scale_without_metadata(tmp_path):
    from fure_defur.validation import validate

    _write_minimal_strand_package(tmp_path)
    adapter_path = tmp_path / "hair_config.yaml"
    adapter = yaml.safe_load(adapter_path.read_text())
    adapter["textured_strands"]["eye_dists_VQA"] = 5.0
    adapter_path.write_text(yaml.safe_dump(adapter, sort_keys=False))

    with pytest.raises(ValueError, match="must be present together"):
        validate(tmp_path)


def test_validate_rejects_adapter_length_mismatch(tmp_path):
    from fure_defur.validation import validate

    _write_minimal_strand_package(tmp_path)
    adapter_path = tmp_path / "hair_config.yaml"
    adapter = yaml.safe_load(adapter_path.read_text())
    adapter["textured_strands"]["mapping_length"]["body"] = .25
    adapter_path.write_text(yaml.safe_dump(adapter, sort_keys=False))

    with pytest.raises(ValueError, match="mapping_length disagrees"):
        validate(tmp_path)


def test_validate_rejects_scale_path_and_missing_scale_basis(tmp_path):
    from fure_defur.io import write_json
    from fure_defur.validation import validate

    _write_minimal_strand_package(tmp_path)
    adapter_path = tmp_path / "hair_config.yaml"
    adapter = yaml.safe_load(adapter_path.read_text())
    adapter["sdf_chamfer"]["path_to_scale"] = "../../outside.pickle"
    adapter_path.write_text(yaml.safe_dump(adapter, sort_keys=False))
    with pytest.raises(ValueError, match="path_to_scale must be empty"):
        validate(tmp_path)

    _write_minimal_strand_package(tmp_path)
    metadata_path = tmp_path / "strand_package.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["scale_basis"] = "unspecified"
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="scale_basis"):
        validate(tmp_path)


def test_end_to_end_synthetic_strand_package(tmp_path):
    from fure_defur.pipeline import run
    from fure_defur.validation import validate
    from fure_defur.io import write_json
    import open3d as o3d
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "vendor/Directional/include").exists():
        pytest.skip("Fetch Directional to run the full CPU smoke test")
    model = trimesh.creation.icosphere(subdivisions=2)
    v, f = model.vertices, model.faces
    model.export(tmp_path / "outer.obj")
    n, edges = g.normals_and_edges(v, f)
    width = np.full(len(v), .035)
    np.savez(tmp_path / "shell.npz", verts=v, faces=f, normals=n, shell_width=width,
             inner_dist=-width, outer_dist=np.zeros(len(v)), reliability_mask_init=np.ones(len(v), bool))
    write_json(tmp_path / "annotations.json", dict(body=list(range(len(v))), empty_part=[]))
    views = tmp_path / "views"
    (views / "images_2").mkdir(parents=True)
    (views / "foreground").mkdir()
    scene = ray_scene(v, f)
    k = np.array([[50., 0, 24], [0, 50., 24], [0, 0, 1]])
    projections = []
    for i, center in enumerate(([0, 0, -3.], [3., 0, 0], [0, 0, 3.])):
        forward = -np.array(center)/3
        right = np.cross([0, 1, 0], forward)
        up = np.cross(forward, right)
        r = np.stack([right, up, forward])
        extr = np.eye(4)
        extr[:3, :3], extr[:3, 3] = r, -r @ center
        p = np.eye(4)
        p[:3] = k @ extr[:3]
        projections.append(p)
        rays = scene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(extr), 48, 48)
        mask = np.isfinite(scene.cast_rays(rays)["t_hit"].numpy()).astype(np.uint8)*255
        Image.fromarray(mask).save(views / "foreground" / f"{i:04d}.png")
        rgb = np.repeat(mask[..., None], 3, axis=-1)
        Image.fromarray(rgb).save(views / "images_2" / f"{i:04d}.png")
    np.save(views / "projection.npy", projections)
    config = dict(schema_version=1, outer_mesh="outer.obj", annotation_mesh="outer.obj", annotations="annotations.json",
                  shell_stats="shell.npz", views="views", cm_per_scene_unit=10., sdf_resolution=24,
                  annotation_origin="transferred from an approved segmented 3D model",
                  part_labels_approved=True,
                  scale_basis="synthetic exact scale")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    status = run(cfg, tmp_path / "result", "shape")
    assert status["ready_for_strand_data"]
    assert status["technical_validation_passed"]
    assert status["annotation_transfer_method"] == "same_topology"
    assert status["part_labels_approved"]
    assert status["downstream_training_scope"] == "external"
    report = validate(tmp_path / "result")
    assert report["views"] == 3
    assert report["tangent_residual_sign_flips"] == 0
    with pytest.raises(FileExistsError):
        run(cfg, tmp_path / "result", "shape")
