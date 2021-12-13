import pickle
from io import StringIO
from typing import List
import boto3
import joblib
import pandas as pd

from nlp_annotation import aws_resources
from nlp_annotation.config.core import config


def create_concatenated_dataframe_from_s3(*, bucket: str, key: str, files_list: List) -> pd.DataFrame:
    """
    Creates a concatenated dataframe from a collection of .csv from a AWS S3 bucket.
    Parameters
    ----------
    bucket: Bucket name.
    key: The file location.
    files_list: The file names list.
    Returns
    -------

    """
    dataframe_list = []

    for file in files_list:
        data_from_s3 = create_dataframe_from_s3(bucket=bucket, key=key + "/" + file)
        dataframe_list.append(data_from_s3)
        dataframe = pd.concat(dataframe_list)
        dataframe = dataframe.drop_duplicates(subset=[config.app_config.dropping_subset])

        return dataframe


def create_dataframe_from_s3(*, bucket: str, key: str) -> pd.DataFrame:
    """
    Return a dataframe from a csv stored in AWS S3.
    Parameters
    ----------
    bucket: Bucket name.
    key: The file location.
    Returns
    -------
    A pandas dataframe from a csv.
    """
    obj = aws_resources.s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    csv_string = body.read().decode("utf-8")

    dataframe = pd.read_csv(StringIO(csv_string))
    return dataframe


def save_dataframe_to_s3(*, dataframe: pd.DataFrame, bucket: str, key: str) -> None:
    """
    Save dataframe into AWS S3 as .csv.
    Parameters
    ----------
    dataframe (pd.Dataframe): The dataframe to be saved.
    bucket (str): Bucket name to save the dataframe.
    key (str): The file location.
    Returns
    -------
    """

    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer)
    s3_resource = aws_resources.s3_resource
    s3_resource.Object(bucket, key).put(Body=csv_buffer.getvalue())


def save_model_to_s3(*, model, bucket: str, key: str) -> None:
    """
    Saves a model as pickle in AWS S3.
    Parameters
    ----------
    model: The model to be saved.
    bucket (str): Bucket name to save the dataframe.
    key (str): The file name.
    Returns
    -------
    """
    pickle_byte_obj = pickle.dumps(model)
    s3_resource = aws_resources.s3_resource
    s3_resource.Object(bucket, key).put(Body=pickle_byte_obj)


def load_model_from_s3(*, bucket: str, key: str):
    """
    Loads the model pickle from AWS S3.
    Parameters
    ----------
    bucket (str): Bucket name to save the dataframe.
    key (str): The file name.
    Returns
    -------
    """

    s3_resource = aws_resources.s3_resource
    model = pickle.loads(s3_resource.Bucket(bucket).Object(key).get()["Body"].read())

    return model