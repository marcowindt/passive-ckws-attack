"""Everything needed to extract Enron and Apache email datasets.
"""

import email
import glob
import mailbox
import os
from typing import Tuple

import colorlog
import pandas as pd
import dask.dataframe as dd
import tensorflow as tf
import tqdm

from ckws_adapted_score_attack.src.config import DATA_TYPE

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")


def split_df(d_frame: pd.DataFrame, frac: float = 0.5):
    first_split = d_frame.sample(frac=frac)
    second_split = d_frame.drop(first_split.index)
    return first_split, second_split


def deterministic_split_df(d_frame: pd.DataFrame, frac: float = 0.5):
    first_split = d_frame[:int(len(d_frame) * frac)]
    second_split = d_frame[int(len(d_frame) * frac):]
    return first_split, second_split


def get_body_from_enron_email(mail):
    """To get the content from raw email"""
    msg = email.message_from_string(mail)
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "".join(parts)


def get_body_from_mboxmsg(msg):
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    body = "".join(parts)
    body = body.split("To unsubscribe")[0]
    return body


def extract_sent_mail_contents(maildir_directory="./maildir/", prefix: str = "") -> pd.DataFrame:
    maildir_directory = prefix + maildir_directory
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*/_sent_mail/*")

    mail_contents = []
    for mailfile_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        with open(mailfile_path, "r") as mailfile:
            raw_mail = mailfile.read()
            mail_contents.append(get_body_from_enron_email(raw_mail))

    return pd.DataFrame(data={"filename": mails, "mail_body": mail_contents})


def extract_apache_ml(maildir_directory="./apache_ml/", prefix: str = "") -> pd.DataFrame:
    maildir_directory = prefix + maildir_directory
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*")
    mail_contents = []
    mail_ids = []
    for mbox_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        for mail in mailbox.mbox(mbox_path):
            mail_content = get_body_from_mboxmsg(mail)
            mail_contents.append(mail_content)
            mail_ids.append(mail["Message-ID"])
    return pd.DataFrame(data={"filename": mail_ids, "mail_body": mail_contents})


def apache_reduced(maildir_directory="./apache_ml/", prefix: str = ""):
    ratio = 30109 / 50878
    apache_full = extract_apache_ml(maildir_directory=maildir_directory, prefix=prefix)
    apache_red, _ = split_df(apache_full, ratio)
    return apache_red


def load_enron(
        file: str = f"extract_kws_documents_dataset-enron-1618238655.7093177.csv",
        path_prefix: str = ''
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    DEPRECATED:
    Function used to load non-sparse tensors from csv file, it will load but it way too large >7GB

    """
    logger.debug(f"Load dataset {path_prefix}{file}")

    # Large enough sample size such that one row fits
    d_frame = dd.read_csv(f"{path_prefix}{file}", sep=';', sample=10000000)
    logger.debug(f"Loaded dataset -> convert to Tensors")

    documents = tf.convert_to_tensor(d_frame.drop(columns=['documents']), dtype=tf.string)
    keywords = tf.convert_to_tensor(d_frame.columns, dtype=tf.string)
    occurrence_array = tf.convert_to_tensor(d_frame.values, dtype=DATA_TYPE)

    logger.debug(f"Converted dataset to Tensors => (occurrence_array, documents, keyword)")

    return occurrence_array, documents, keywords


def load_preprocessed_enron(
    path: str = f"./extract_kws_documents_dataset-enron",
    prefix: str = "",
    dense_shape: tuple = (30109, 62976),
) -> Tuple[tf.SparseTensor, tf.Tensor, tf.Tensor]:
    return load_preprocessed_dataset(path=path, prefix=prefix, dense_shape=dense_shape)


def load_preprocessed_enron_float32(
    path: str = f"./extracted_kws_documents_dataset_float32-enron",
    prefix: str = "",
    dense_shape: tuple = (30109, 62976),
) -> Tuple[tf.SparseTensor, tf.Tensor, tf.Tensor]:
    return load_preprocessed_dataset(path=path, prefix=prefix, dense_shape=dense_shape)


def load_preprocessed_apache(
    path: str = f"./extract_kws_documents_dataset-apache",
    prefix: str = "",
    dense_shape: tuple = (50581, 92475),
) -> Tuple[tf.SparseTensor, tf.Tensor, tf.Tensor]:
    return load_preprocessed_dataset(path=path, prefix=prefix, dense_shape=dense_shape)


def load_preprocessed_apache_reduced(
    path: str = f"./extract_kws_documents_dataset-apache_reduced",
    prefix: str = "",
    dense_shape: tuple = (29974, 75091),
) -> Tuple[tf.SparseTensor, tf.Tensor, tf.Tensor]:
    return load_preprocessed_dataset(path=path, prefix=prefix, dense_shape=dense_shape)


def load_preprocessed_dataset(
        path: str = f"./extract_kws_documents_dataset-enron",
        prefix: str = "",
        dense_shape: tuple = (30109, 62976),
) -> Tuple[tf.SparseTensor, tf.Tensor, tf.Tensor]:
    """
    Load preprocessed datasets.
    Will load sparse tensors: document-keyword frequency, global keyword vocabulary, global keyword occurrences
    """
    logger.info(f"Load preprocessed dataset at: {prefix}{path}")

    logger.debug(f"Load {prefix}{path}")
    document_keyword_occurrences = tf.data.experimental.load(
        path=f"{prefix}{path}",
        element_spec=tf.SparseTensorSpec(shape=dense_shape, dtype=DATA_TYPE)
    )
    document_keyword_occurrence: tf.SparseTensor = _unpack_dataset(document_keyword_occurrences)

    logger.debug(f"Load {prefix}{path}-voc")
    sorted_keywords_voc = tf.data.experimental.load(
        path=f"{prefix}{path}-voc",
        element_spec=tf.TensorSpec(shape=(dense_shape[1],), dtype=tf.string)
    )
    sorted_keyword_voc: tf.Tensor = _unpack_dataset(sorted_keywords_voc)

    logger.debug(f"Load {prefix}{path}-occ")
    sorted_keywords_occ = tf.data.experimental.load(
        path=f"{prefix}{path}-occ",
        element_spec=tf.TensorSpec(shape=(dense_shape[1],), dtype=tf.int64)
    )
    sorted_keyword_occ: tf.Tensor = _unpack_dataset(sorted_keywords_occ)

    return document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ


def _unpack_dataset(datasets):
    """
    tf dataset is capable of multiple datasets, but we only stored one, so we need an ugly unpack function
    """
    datasets = [dataset for dataset in datasets]
    return datasets[0]
