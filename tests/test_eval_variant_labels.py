from pathlib import Path

import yaml

import eval_homeo_mem


def train_run(path: Path, seed: int, shuffle: bool, lr: float) -> Path:
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"seed": seed, "shuffle_control": shuffle, "lr": lr}))
    return path


def test_variant_labels_disambiguate_same_seed_runs_by_lr(tmp_path):
    a = train_run(tmp_path / "20261118-191035_Qwen", 21, False, 2e-5)
    b = train_run(tmp_path / "20261118-233022_Qwen", 21, False, 1e-4)
    s = train_run(tmp_path / "20261118-213827_Qwen", 21, True, 2e-5)
    assert eval_homeo_mem.variant_labels([a, b, s]) == ["trained_s21_lr2e-05", "trained_s21_lr0.0001", "shuffle_s21"]


def test_variant_labels_unchanged_without_collision(tmp_path):
    a = train_run(tmp_path / "a", 21, False, 2e-5)
    s = train_run(tmp_path / "s", 21, True, 2e-5)
    assert eval_homeo_mem.variant_labels([a, s]) == ["trained_s21", "shuffle_s21"]
