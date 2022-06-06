import argparse

import colorlog
import logging

from ckws_adapted_score_attack.src.common import setup_logger
from ckws_adapted_score_attack.src.preprocess import DatasetPreprocessor

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")


def preprocess(*args, **kwargs):
    extracted_dataset_name = kwargs.get("extracted_dataset_name", "extracted_kws_documents_dataset_float32-enron")
    src_path_prefix = kwargs.get("src_path_prefix", "../")
    save_path_prefix = kwargs.get("save_path_prefix", "../")

    setup_logger()
    FORMATTER = logging.Formatter("[%(asctime)s %(levelname)s] %(module)s: %(message)s")
    file_handler = logging.FileHandler("preprocess.log")
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)

    preprocessor = DatasetPreprocessor(
        # dataset='apache_reduced',
        path_prefix=src_path_prefix,
        build_occurrence_array=True,
        sparse=True,
        extracted_dataset_name=extracted_dataset_name,
    )
    preprocessor.save_serialized_occ_array(save_path_prefix=save_path_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--extracted_dataset_name",
        type=str,
        default="extracted_kws_documents_dataset_float32-enron",
        help="Name of the folder to store preprocessed dataset.",
    )
    parser.add_argument(
        "--src-path-prefix",
        type=str,
        default="../",
        help="Path prefix for preprocessor to find raw dataset.",
    )
    parser.add_argument(
        "--save-path-prefix",
        type=str,
        default="./",
        help="Path prefix to store the preprocessed dataset.",
    )

    params = parser.parse_args()
    preprocess(**vars(params))
