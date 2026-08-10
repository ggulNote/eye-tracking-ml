from pathlib import Path

from PIL import Image
from scipy.io import savemat

from gaze_pipeline.data.mpiifacegaze import read_mpiifacegaze_dataset


def write_subject(root: Path, subject_id: str = "p00") -> Path:
    subject = root / subject_id
    image_path = subject / "day01" / "0001.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (128, 72), color=(0, 0, 0)).save(image_path)

    calibration = subject / "Calibration" / "screenSize.mat"
    calibration.parent.mkdir(parents=True)
    savemat(
        calibration,
        {
            "width_pixel": 1440,
            "height_pixel": 900,
            "width_mm": 286.4,
            "height_mm": 179.0,
        },
    )

    numeric = [
        720,
        450,
        40,
        25,
        50,
        25,
        75,
        25,
        85,
        25,
        50,
        50,
        75,
        50,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        500.0,
        0.0,
        0.0,
        500.0,
        0.0,
        0.0,
        0.0,
    ]
    assert len(numeric) == 26
    row = " ".join(["day01/0001.jpg", *(str(value) for value in numeric), "right"])
    (subject / f"{subject_id}.txt").write_text(f"{row}\n", encoding="utf-8")
    return subject


def test_reads_one_static_image_annotation_and_calibration(tmp_path: Path) -> None:
    write_subject(tmp_path)

    records = read_mpiifacegaze_dataset(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.sample_id == "p00:day01/0001.jpg"
    assert record.view == "front"
    assert record.pair_id is None
    assert record.image_width_px == 128
    assert record.image_height_px == 72
    assert record.screen.width_px == 1440
    assert record.screen.height_px == 900
    assert record.gaze_screen_xy_normalized == (0.0, 0.0)
    assert len(record.facial_landmarks_xy) == 6
