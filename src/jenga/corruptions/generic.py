from __future__ import annotations

import numpy as np
import pandas as pd

from ..basis import DataCorruption, TabularCorruption


# Inject different kinds of missing values
class MissingValues(TabularCorruption):

    def __init__(
        self,
        column: str | int,
        fraction: float,
        na_value: float = np.nan,
        missingness: str = "MCAR_COL",
        seed: int | float | None = None,
        deps: list | None = None,
    ):
        """
        Missing value corruptions for structured data.

        Args:
            column (str | int):         The identifier of the column to pollute with missing values
            fraction (float):           The fraction of rows to corrupt. Must be between 0 and 1.
            na_value (float, optional): The value that represents a missing value, defaults to :any:`numpy.nan`
            missingness (str):          The sampling mechanism used for the missing values.
                                        Must be a value of
                                        ['MCAR_COL', 'MAR_RAND', 'MNAR_SELF', 'MCAR_TAB', 'MAR_DK', 'MNAR_DK'].
                                        Deprecated values: ['MCAR', 'MAR', 'MNAR'].
                                        Defaults to `MCAR_COL`.
            seed (int | float | None, optional): The seed for the random operations.
                                        Defaults to :any:`None`, which means no seed is used.
            deps (list):                A list of columns the selected :paramref:`column` depends on.
                                        Required for sampling mechanism that involve domain-knowledge.
                                        TODO: Set?

        Raises:
            ValueError:  If :paramref:`fraction` is not between 0 and 1, or if :paramref:`missingness` contains an
                         unsupported value.
        """
        super().__init__(column, fraction, sampling=missingness)

        self.na_value: float = na_value
        self.seed: int | float | None = seed
        self.deps: list = deps

    def transform(self, data: pd.DataFrame):
        corrupted_data: pd.DataFrame = data.copy(deep=True)
        if self.sampling in [
            "MCAR",
            "MAR",
            "MNAR",
            "MCAR_COL",
            "MAR_RAND",
            "MNAR_SELF",
        ]:
            if self.sampling == "MCAR_COL":
                self.sampling = "MCAR"
            elif self.sampling == "MAR_RAND":
                self.sampling = "MAR"
            elif self.sampling == "MNAR_SELF":
                self.sampling = "MNAR"
            # Replace the values in the selected column (self.column) by self.na_value in the records specified in rows.
            rows: pd.Index = self.sample_rows(corrupted_data, self.seed)
            corrupted_data.loc[rows, [self.column]] = self.na_value
        elif self.sampling == "MCAR_TAB":
            value_list: np.ndarray = corrupted_data.to_numpy().flatten()
            all_indices: np.ndarray = np.arange(len(value_list))
            if self.seed is not None:
                np.random.seed(self.seed)
            indices_to_modify: np.ndarray = np.random.permutation(all_indices)[
                : int(len(all_indices) * self.fraction)
            ]
            value_list[indices_to_modify] = self.na_value
            corrupted_values = value_list.reshape(corrupted_data.shape)
            corrupted_data = pd.DataFrame(
                corrupted_values, corrupted_data.index, corrupted_data.columns
            )
        elif self.sampling in ["MAR_DK", "MNAR_DK"]:
            deps = self.deps
            if deps is None:
                raise ValueError(
                    "No list of dependencies was provided for the domain-knowledge based introduction of missing values."
                )
            elif len(deps) > 0:
                if self.sampling == "MNAR_DK":
                    deps.insert(0, self.column)
                if self.seed is not None:
                    np.random.seed(self.seed)
                n_values_to_discard = int(len(data) * min(self.fraction, 1.0))
                perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
                perc_idx = range(
                    perc_lower_start, perc_lower_start + n_values_to_discard
                )

                # pick a random percentile of values in other column based on which the rows will be polluted
                rows = data[deps].sort_values(deps).iloc[perc_idx].index
                corrupted_data.loc[rows, [self.column]] = self.na_value

        return corrupted_data


# Missing Values based on the records' "difficulty" for the model
class MissingValuesBasedOnEntropy(DataCorruption):

    def __init__(
        self, column, fraction, most_confident, model, data_to_predict_on, na_value
    ):
        self.column = column
        self.fraction = fraction
        self.most_confident = most_confident
        self.model = model
        self.data_to_predict_on = data_to_predict_on
        self.na_value = na_value

        super().__init__()

    def transform(self, data):
        df = data.copy(deep=True)

        cutoff = int(len(df) * (1 - self.fraction))
        probas = self.model.predict_proba(self.data_to_predict_on)

        if self.most_confident:
            affected = probas.max(axis=1).argsort()[:cutoff]

        else:
            # for samples with the smallest maximum probability the model is most uncertain
            affected = probas.max(axis=1).argsort()[-cutoff:]

        df.loc[df.index[affected], self.column] = self.na_value

        return df


# Swapping a fraction of the values between two columns, mimics input errors in forms
# and programming errors during data preparation
class SwappedValues(TabularCorruption):

    def __init__(self, column, fraction, sampling="CAR", swap_with=None):
        super().__init__(column, fraction, sampling)
        self.swap_with = swap_with

    def transform(self, data):
        df = data.copy(deep=True)
        if not self.swap_with:
            self.swap_with = np.random.choice(
                [c for c in data.columns if c != self.column]
            )

        rows = self.sample_rows(df)

        tmp_vals = df.loc[rows, self.swap_with].copy(deep=True)
        df.loc[rows, self.swap_with] = df.loc[rows, self.column]
        df.loc[rows, self.column] = tmp_vals

        return df


class CategoricalShift(TabularCorruption):
    def transform(self, data):
        df = data.copy(deep=True)
        rows = self.sample_rows(df)
        numeric_cols, non_numeric_cols = self.get_dtype(df)

        if self.column in numeric_cols:
            print("CategoricalShift implemented only for categorical variables")
            return df

        else:
            histogram = df[self.column].value_counts()
            random_other_val = np.random.permutation(histogram.index)
            df.loc[rows, self.column] = df.loc[rows, self.column].replace(
                histogram.index, random_other_val
            )
            return df
