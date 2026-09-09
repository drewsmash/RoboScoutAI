from ramscout.field import ALLIANCE_DEPTH, FIELD_LENGTH, FIELD_WIDTH, zone_name
from ramscout.geometry import default_source_points, homography_from_corners, project_points


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
