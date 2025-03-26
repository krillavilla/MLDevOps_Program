#!/usr/bin/env python
"""
This script cleans the input data
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Clean the data
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Read the data
    logger.info("Reading data from artifact")
    df = pd.read_csv(artifact_path)

    # Drop outliers
    logger.info("Cleaning data")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save the cleaned data
    logger.info("Saving cleaned data")
    df.to_csv("clean_sample.csv", index=False)

    # Upload the cleaned data as an artifact
    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for cleaning",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for cleaning",
        required=True
    )

    args = parser.parse_args()

    go(args)