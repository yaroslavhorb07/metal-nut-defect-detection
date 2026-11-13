from PIL import Image

from nutdetect.data import build_manifest, split_manifest


def make_folder(root, n_good, n_bad):
    for name, n in (("good", n_good), ("bad", n_bad)):
        d = root / name
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (16, 16)).save(d / f"{i}.png")


def test_manifest_labels(tmp_path):
    make_folder(tmp_path, 6, 3)
    df = build_manifest(tmp_path)
    assert df["label"].sum() == 3
    assert len(df) == 9


def test_split_is_stratified_and_disjoint(tmp_path):
    make_folder(tmp_path, 40, 20)
    df = split_manifest(build_manifest(tmp_path), 0.15, 0.15, seed=1)
    assert df["path"].is_unique
    for part in ("train", "val", "test"):
        share = df[df["split"] == part]["label"].mean()
        assert abs(share - 1 / 3) < 0.1
    assert (df["split"] == "test").sum() >= 8


def test_split_is_reproducible(tmp_path):
    make_folder(tmp_path, 30, 15)
    df = build_manifest(tmp_path)
    a = split_manifest(df, 0.15, 0.15, seed=7)
    b = split_manifest(df, 0.15, 0.15, seed=7)
    assert a["split"].tolist() == b["split"].tolist()
