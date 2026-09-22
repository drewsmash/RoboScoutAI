from ramscout.field import ALLIANCE_DEPTH, FIELD_LENGTH, FIELD_WIDTH, zone_name
from ramscout.geometry import default_source_points, homography_from_corners, project_points, reproject_samples


def test_zone_names_split_the_field():
    assert zone_name(20, FIELD_WIDTH / 2).startswith("blue")
    assert zone_name(FIELD_LENGTH - 20, FIELD_WIDTH / 2).startswith("red")
    assert zone_name(FIELD_LENGTH / 2, FIELD_WIDTH / 2) == "neutral"
    assert "hub" in zone_name(ALLIANCE_DEPTH, FIELD_WIDTH / 2)


def test_default_homography_maps_corners_to_field():
    src = default_source_points(1920, 1080)
    H = homography_from_corners(src)
    mapped = project_points(src.tolist(), H)
    assert mapped[0][0] < 5
    assert mapped[0][1] < 5
    assert abs(mapped[1][0] - FIELD_LENGTH) < 5
    assert abs(mapped[2][1] - FIELD_WIDTH) < 5


def test_reproject_samples_maps_pixel_feet():
    src = default_source_points(1920, 1080)
    samples = [{"px": float(src[0][0]), "py": float(src[0][1]), "x": 99.0, "y": 99.0}]
    out = reproject_samples(samples, src.tolist())
    assert out[0]["field_valid"] is True
    assert out[0]["x"] < 5
    assert out[0]["y"] < 5
    corner = [{"px": float(src[1][0]), "py": float(src[1][1]), "x": 0.0, "y": 0.0}]
    out2 = reproject_samples(corner, src.tolist())
    assert out2[0]["field_valid"] is True
    assert abs(out2[0]["x"] - FIELD_LENGTH) < 8


def test_reproject_samples_marks_outside_unavailable():
    src = default_source_points(1920, 1080)
    samples = [{"px": 1.0, "py": 1.0}]
    out = reproject_samples(samples, src.tolist())
    assert out[0]["field_valid"] is False
    assert out[0]["x"] is None
    assert out[0]["y"] is None
    # Explicitly not clamped to the (0,0) corner as a valid pose.
    assert not (out[0].get("x") == 0 and out[0].get("y") == 0 and out[0].get("field_valid"))
