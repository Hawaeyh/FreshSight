import pytest
from ai.dataset import inspect_dataset, CLASS_NAMES


def test_inspect_dataset_finds_required_classes():
    df = inspect_dataset()
    assert all(class_name in df["class_name"].unique() for class_name in CLASS_NAMES)
    assert not df.empty


def test_inspect_dataset_no_unsupported_class():
    df = inspect_dataset()
    unsupported = [name for name in df["class_name"].unique() if name not in CLASS_NAMES]
    assert unsupported == []
