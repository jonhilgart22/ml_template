# Module to build model with Sagemaker

import json
from typing import Optional

import click
from sagemaker.estimator import Estimator

from predict_cpt_codes.utilities.log_utils import setup_logging


@click.command()
@click.option(
    "--image-name",
    required=True,
    default="ml-suggest-billing-codes-training:latest",
    help="name of image that conforms to sagemaker training expectations",
)
@click.option("--instance-type", default="local", help="Instance type to train on, ex. 'local', 'ml.c5.2xlarge")
@click.option("--instance-count", default=1, help="number of instances to use for training")
@click.option(
    "--train-volume-size", default=30, help="Size (GB) of the EBS volume mounted to instance for storing training data"
)
@click.option(
    "--s3-prefix",
    default="s3://onemedical-ml-sagemaker/ml-suggest-billing-codes",
    help="The s3 location for input and output data",
)
@click.option(
    "--hyperparameter-file",
    default="predict_cpt_codes/modeling/sagemaker/hyperparameters.json",
    help="file containing desired input arguments to standard.train module",
)
@click.option("--train-directory", default=None, help="path beyond s3 prefix (if any) to training data")
def main(
    image_name: str,
    instance_type: str,
    instance_count: int,
    train_volume_size: int,
    s3_prefix: str,
    hyperparameter_file: Optional[str],
    train_directory: str,
) -> None:
    """Main function for executing sagemaker build routine"""
    setup_logging()
    # Configuring read/write paths in S3
    output_path = f"{s3_prefix}/output/"
    input_path = f"{s3_prefix}/input/data/training/"
    input_path += train_directory if train_directory is not None else ""
    # Loading in model hyperparameters, if one is provided
    if hyperparameter_file is not None:
        params = json.load(open(hyperparameter_file, "r"))
    else:
        params = {}
    params.update({"s3-input-path": input_path})
    # Creating the sagemaker estimator
    estimator_kwargs = {
        "image_name": image_name,
        "role": "ml-sagemaker-training",
        "train_instance_count": instance_count,
        "train_instance_type": instance_type,
        "train_volume_size": train_volume_size,
        "output_path": output_path,
        "hyperparameters": params,
        "encrypt_inter_container_traffic": True,
        "subnets": ["subnet-0fb66fe8906f43f2c", "subnet-0fb66fe8906f43f2c"],  # subnets for ml flow
        "security_group_ids": ["sg-08a4677231f689a4b"],  # security group for alb for mlflow production
    }
    sage = Estimator(**estimator_kwargs)
    # Fitting the estimator w/ either local or remote (s3) data
    input_dict = {"training": input_path}
    sage.fit(input_dict)


if __name__ == "__main__":
    main()