from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_annotation import data_manager, pipeline, utils
from nlp_annotation.config.core import config


def train_model(train_features, bucket: str, key: str) -> Dict:
    """
    Trains the model.
    Parameters
    ----------
    features_dict (Dict): A dict with the features set.
    Returns
    -------
    """
    features_dict = create_features_dict(train_features)

    X_train = features_dict[config.app_config.x_train]
    X_test = features_dict[config.app_config.x_test]
    y_train = features_dict[config.app_config.y_train]
    y_test = features_dict[config.app_config.y_test]
    model = pipeline.model_pipeline

    model.fit(X_train, y_train)

    train_y_pred = model.predict(X_train)
    test_y_pred = model.predict(X_test)

    train_accuracy = utils.calculate_accuracy(
        real_values=y_train,
        predicted_values=train_y_pred
        )

    test_accuracy = utils.calculate_accuracy(
        real_values=y_test,
        predicted_values=test_y_pred
        )

    print("Train Accuracy    :", train_accuracy)
    print("Test Accuracy    :", test_accuracy)

    features_dict[config.app_config.model_key] = model

    data_manager.save_model_to_s3(model=model, bucket=bucket, key=key)

    return features_dict


def create_features_dict(train_features: pd.DataFrame) -> Dict:
    """
    Creates a dict to hold the data splits and later types of objects from the train step.
    Parameters
    ----------
    train_features (pd.DataFrame): The predictor variables.
    target (pd.DataFrame): The target variable.
    Returns
    -------
    """
    features = train_features[config.model_config.train_features_column]
    target = train_features[config.model_config.target]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.model_config.split_test_size,
        random_state=config.model_config.random_state,
    )

    features_dict = {
        config.app_config.x_train: X_train,
        config.app_config.x_test: X_test,
        config.app_config.y_train: y_train,
        config.app_config.y_test: y_test,
    }

    return features_dict
