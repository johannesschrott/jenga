import numpy as np
import pandas as pd
import pytest

from jenga.corruptions.generic import MissingValues

SEED = 0xCAFE


@pytest.fixture
def dataFrame_10x10() -> pd.DataFrame:
    df = pd.DataFrame(np.zeros((10, 10)))
    assert df.shape == (10, 10)
    return df


def test_mcar_col(dataFrame_10x10: pd.DataFrame):
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MCAR", seed=SEED)
        .transform(dataFrame_10x10)
        .equals(
            MissingValues(
                column=1, fraction=0.5, missingness="MCAR_COL", seed=SEED
            ).transform(dataFrame_10x10)
        )
    )
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MCAR_COL", seed=SEED)
        .transform(dataFrame_10x10)
        .isna()
        .sum()
        .sum()
        == 5
    )


def test_mcar_tab(dataFrame_10x10: pd.DataFrame):
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MCAR_TAB", seed=SEED)
        .transform(dataFrame_10x10)
        .equals(
            MissingValues(
                column=1, fraction=0.5, missingness="MCAR_TAB", seed=SEED
            ).transform(dataFrame_10x10)
        )
    )
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MCAR_TAB", seed=SEED)
        .transform(dataFrame_10x10)
        .isna()
        .sum()
        .sum()
        == 50
    )


def test_mar_rand(dataFrame_10x10: pd.DataFrame):
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MAR", seed=SEED)
        .transform(dataFrame_10x10)
        .equals(
            MissingValues(
                column=1, fraction=0.5, missingness="MAR_RAND", seed=SEED
            ).transform(dataFrame_10x10)
        )
    )
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MAR_RAND", seed=SEED)
        .transform(dataFrame_10x10)
        .isna()
        .sum()
        .sum()
        == 5
    )


def test_mar_dk(dataFrame_10x10: pd.DataFrame):
    assert (
        MissingValues(
            column=1, fraction=0.5, missingness="MAR_DK", seed=SEED, deps=[2, 4, 6]
        )
        .transform(dataFrame_10x10)
        .equals(
            MissingValues(
                column=1, fraction=0.5, missingness="MAR_DK", seed=SEED, deps=[2, 4, 6]
            ).transform(dataFrame_10x10)
        )
    )
    assert (
        MissingValues(
            column=1, fraction=0.5, missingness="MAR_DK", seed=SEED, deps=[2, 4, 6]
        )
        .transform(dataFrame_10x10)
        .isna()
        .sum()
        .sum()
        == 5
    )


def test_mnar_rand(dataFrame_10x10: pd.DataFrame):
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MNAR", seed=SEED)
        .transform(dataFrame_10x10)
        .equals(
            MissingValues(
                column=1, fraction=0.5, missingness="MNAR_SELF", seed=SEED
            ).transform(dataFrame_10x10)
        )
    )
    assert (
        MissingValues(column=1, fraction=0.5, missingness="MNAR_SELF", seed=SEED)
        .transform(dataFrame_10x10)
        .isna()
        .sum()
        .sum()
        == 5
    )


def test_mnar_dk(dataFrame_10x10: pd.DataFrame):
    assert (
        MissingValues(
            column=1, fraction=0.5, missingness="MNAR_DK", seed=SEED, deps=[2, 4, 6]
        )
        .transform(dataFrame_10x10)
        .equals(
            MissingValues(
                column=1, fraction=0.5, missingness="MNAR_DK", seed=SEED, deps=[2, 4, 6]
            ).transform(dataFrame_10x10)
        )
    )
    assert (
        MissingValues(
            column=1, fraction=0.5, missingness="MNAR_DK", seed=SEED, deps=[2, 4, 6]
        )
        .transform(dataFrame_10x10)
        .isna()
        .sum()
        .sum()
        == 5
    )
