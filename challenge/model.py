from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


SCHEDULED_DATE_COL = "Fecha-I"
OPERATION_DATE_COL = "Fecha-O"
TARGET_COL = "delay"
AIRLINE_COL = "OPERA"
FLIGHT_TYPE_COL = "TIPOVUELO"
MONTH_COL = "MES"
DELAY_THRESHOLD_IN_MINUTES = 15


class DelayModel:
    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    MODEL_PATH = Path(__file__).resolve().parent / "delay_model.joblib"

    def __init__(self):
        self._model = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.
        
        The selected top-10 model only uses one-hot encodings from OPERA, MES,
        and TIPOVUELO. The delay target is reconstructed only during training.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data = data.copy()
        self._validate_raw_feature_columns(data)

        features = self._build_features(data)

        if target_column is None:
            return features

        target = self._build_target(
            data=data,
            target_column=target_column,
        )
        return features, target

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._validate_preprocessed_features(features)
        self._validate_target(target)

        target_series = target.squeeze()

        n_y0 = int((target_series == 0).sum())
        n_y1 = int((target_series == 1).sum())
        n_total = int(len(target_series))

        if n_total == 0 or n_y0 == 0 or n_y1 == 0:
            raise ValueError("Target must contain both delay classes.")

        class_weight = {
            0: n_y1 / n_total,
            1: n_y0 / n_total,
        }

        self._model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
        )
        self._model.fit(features, target_series)

    def fit_from_data(
        self,
        data: pd.DataFrame,
        target_column: str = TARGET_COL,
    ) -> None:
        """
        Fit the model directly from raw training data.
        """
        features, target = self.preprocess(
            data=data,
            target_column=target_column,
        )
        self.fit(
            features=features,
            target=target,
        )

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted or loaded.")

        self._validate_preprocessed_features(features)

        predictions = self._model.predict(features)
        return [int(prediction) for prediction in predictions]
    
    def save(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Persist the fitted model to disk.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted.")

        model_path = Path(path) if path is not None else self.MODEL_PATH
        joblib.dump(self._model, model_path)

    def load(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load a persisted model from disk.
        """
        model_path = Path(path) if path is not None else self.MODEL_PATH
        self._model = joblib.load(model_path)

    def _build_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        encoded_features = pd.concat(
            [
                pd.get_dummies(data[AIRLINE_COL], prefix=AIRLINE_COL),
                pd.get_dummies(data[FLIGHT_TYPE_COL], prefix=FLIGHT_TYPE_COL),
                pd.get_dummies(data[MONTH_COL], prefix=MONTH_COL),
            ],
            axis=1,
        )

        return encoded_features.reindex(
            columns=self.FEATURES_COLS,
            fill_value=0,
        )

    def _build_target(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> pd.DataFrame:
        if target_column in data.columns:
            return data[[target_column]]

        self._validate_target_source_columns(data)

        scheduled_dates = pd.to_datetime(data[SCHEDULED_DATE_COL])
        operation_dates = pd.to_datetime(data[OPERATION_DATE_COL])
        min_diff = (operation_dates - scheduled_dates).dt.total_seconds() / 60

        target = (min_diff > DELAY_THRESHOLD_IN_MINUTES).astype(int)
        return pd.DataFrame({target_column: target})

    def _validate_raw_feature_columns(
        self,
        data: pd.DataFrame,
    ) -> None:
        required_columns = {AIRLINE_COL, FLIGHT_TYPE_COL, MONTH_COL}
        missing_columns = required_columns.difference(data.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required feature columns: {missing}")

    def _validate_target_source_columns(
        self,
        data: pd.DataFrame,
    ) -> None:
        required_columns = {SCHEDULED_DATE_COL, OPERATION_DATE_COL}
        missing_columns = required_columns.difference(data.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required target source columns: {missing}")

    def _validate_preprocessed_features(
        self,
        features: pd.DataFrame,
    ) -> None:
        missing_columns = set(self.FEATURES_COLS).difference(features.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing preprocessed feature columns: {missing}")

    def _validate_target(
        self,
        target: pd.DataFrame,
    ) -> None:
        if target.shape[1] != 1:
            raise ValueError("Target must contain exactly one column.")