import multiprocessing
from typing import List

import sqlite3
from sqlite3 import Error

import colorlog
import nltk
import numpy as np
import tensorflow as tf
import pandas as pd

from functools import reduce

import tqdm

from ckws_adapted_score_attack.src.common import KeywordExtractor, increase_occ_array_element_wise
from ckws_adapted_score_attack.src.email_extraction import extract_sent_mail_contents, extract_apache_ml, apache_reduced
from contextlib import contextmanager

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")

MAX_CPUS = multiprocessing.cpu_count() if multiprocessing.cpu_count() <= 12 else 12
DATA_TYPE = tf.float32  # tf.float64

DocumentSetExtraction = {
    "enron": extract_sent_mail_contents,
    "apache": extract_apache_ml,
    "apache_reduced": apache_reduced,
}


@contextmanager
def pool_context(*args, **kwargs):
    """Context manager to standardize the parallelized functions.
    """
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


class OccRowComputerPreprocessor:
    """Callable class used to parallelize occurrence matrix computation
    """

    def __init__(self, sorted_voc_with_occ):
        self.voc = [word for word, global_occ in sorted_voc_with_occ]

    def __call__(self, document_word_occurrence_dict):
        # Set 1 if word in word_list of a document, otherwise 0
        return [document_word_occurrence_dict[voc_word] if voc_word in document_word_occurrence_dict else 0
                for voc_word in self.voc]


class DatasetPreprocessor:

    AVAILABLE_DATASETS = DocumentSetExtraction.keys()

    create_emails_table = """CREATE TABLE IF NOT EXISTS emails (
        id integer PRIMARY KEY,
        email text NOT NULL
    );"""

    create_keywords_table = """CREATE TABLE IF NOT EXISTS keywords (
        id integer PRIMARY KEY,
        keyword text NOT NULL
    );"""

    create_keyword_email_occurrence_table = """CREATE TABLE IF NOT EXISTS occurrence (
        email_id integer,
        keyword_id integer,
        occurrence integer,
        FOREIGN KEY (email_id) REFERENCES emails (id),
        FOREIGN KEY (keyword_id) REFERENCES keywords (id),
        PRIMARY KEY (email_id, keyword_id)
    );"""

    def __init__(
            self,
            dataset: str = 'enron',
            path_prefix: str = '',
            extracted_dataset_name: str = None,
            build_occurrence_array: bool = True,
            sparse: bool = True,
            one_occ_per_doc: bool = True,
            kw_conjunction_size: int = 1,
    ):
        """
        Keyword extraction process, used to prevent redoing keyword extraction in experiments

        Arguments:
            dataset {str}                -- Name of the dataset to extract
            path_prefix {str}            -- Prefix to locate dataset in filesystem
            extracted_dataset_name {str} -- Name of extracted keyword dataset, will be saved with this name
            build_occurrence_array {bool}-- Determines whether or not to create the occurrence array from the extracted
                                            keywords
            sparse {bool}                -- If true occurrence array will be a tf.SparseTensor, otherwise tf.Tensor.
            one_occ_per_doc {bool}       -- If false also take into account the number of occurrence in email
                                            If true then only count if it occurs in an email,
                                            so not the number of occurrences
            kw_conjunction_size {int}     -- Only used to reduce dataset size,
                                            since kws having freq less than kw_conjunction_size won't reappear in the
                                            element wise multiplied matrix during the experiments
        """
        if dataset not in self.AVAILABLE_DATASETS:
            raise ValueError(f'Unknown dataset: {dataset}')

        self.extracted_dataset_name = extracted_dataset_name
        if self.extracted_dataset_name is None:
            self.extracted_dataset_name = f"extract_kws_documents_dataset-{dataset}"

        extraction_procedure = DocumentSetExtraction[dataset]

        # Read all mails into memory
        df_frame = extraction_procedure(prefix=path_prefix)

        with pool_context(processes=MAX_CPUS) as pool:
            results = pool.starmap(
                self.extract_email_voc, zip(
                    range(MAX_CPUS),
                    np.array_split(df_frame, MAX_CPUS),
                    [one_occ_per_doc for _ in range(MAX_CPUS)],   # occurrence_per_doc
                ),
            )
            self.freq_dict, self.glob_freq_dict = reduce(KeywordExtractor._merge_results, results)

        # Remove corpus from memory
        del df_frame

        # Keep all
        self.glob_freq_dict = nltk.FreqDist(self.glob_freq_dict)
        self.glob_freq_list = self.glob_freq_dict.most_common()

        # Sort from highest word count to lowest word count
        self.sorted_voc_with_occ_original = sorted(
            [(word, count) for word, count in self.glob_freq_list if count >= kw_conjunction_size],
            key=lambda d: d[1],  # d[1] a.k.a. count
            reverse=True,
        )

        if build_occurrence_array:
            if not sparse:
                self.original_occ_array = self.build_occurrence_array(
                    sorted_voc_with_occ=self.sorted_voc_with_occ_original,
                    freq_dict=self.freq_dict)
            else:
                self.original_occ_array = self.build_sparse_occurrence_array(
                    sorted_voc_with_occ=self.sorted_voc_with_occ_original,
                    freq_dict=self.freq_dict)

    @staticmethod
    def build_sparse_occurrence_array(sorted_voc_with_occ: List, freq_dict: dict) -> tf.Tensor:
        occ_list = []

        with pool_context(processes=MAX_CPUS) as pool:
            for row in tqdm.tqdm(
                    pool.imap_unordered(OccRowComputerPreprocessor(sorted_voc_with_occ), freq_dict.values()),
                    desc=f"Computing the occurrence array",
                    total=len(freq_dict.values()),
            ):
                occ_list.append(tf.sparse.transpose(tf.sparse.from_dense(tf.convert_to_tensor([row], dtype=tf.float32))))

        return tf.sparse.transpose(tf.sparse.concat(axis=1, sp_inputs=occ_list))

    @staticmethod
    def build_occurrence_array(sorted_voc_with_occ: List, freq_dict: dict) -> tf.Tensor:
        occ_list = []
        with pool_context(processes=MAX_CPUS) as pool:
            for row in tqdm.tqdm(
                    pool.imap_unordered(OccRowComputerPreprocessor(sorted_voc_with_occ), freq_dict.values()),
                    desc=f"Computing the occurrence array",
                    total=len(freq_dict.values()),
            ):
                occ_list.append(row)

        return tf.convert_to_tensor(occ_list, dtype=tf.float32)

    def save_to_csv(self):
        if self.original_occ_array is None:
            raise ValueError("self.original_occ_array is None")

        d_frame = pd.DataFrame(
            self.original_occ_array.numpy(),
            columns=[kw for kw, _occ in self.sorted_voc_with_occ_original]
        )
        d_frame.insert(loc=0, column='documents', value=list(self.freq_dict.keys()), allow_duplicates=True)
        d_frame.to_csv(f"{self.extracted_dataset_name}.csv", sep=';')

    def save_serialized_occ_array(self, save_path_prefix: str = ""):
        if self.original_occ_array is None:
            raise ValueError("self.original_occ_array is None")

        logger.debug(f"Creating dataset to save")
        dataset = tf.data.Dataset.from_tensors(self.original_occ_array)

        logger.info(f"self.original_occ_array shape: {self.original_occ_array.shape}")
        tf.data.experimental.save(dataset, f"{save_path_prefix}{self.extracted_dataset_name}")
        logger.info(f"Dataset saved at: {save_path_prefix}{self.extracted_dataset_name}")
        print(f"Occurrence array serialized and saved at: {save_path_prefix}{self.extracted_dataset_name}")
        print(f"Original occ array shape: {self.original_occ_array.shape}")

        logger.info(f"Creating sorted_voc_dataset")
        voc = tf.convert_to_tensor([word for word, _occ in self.sorted_voc_with_occ_original])
        occ = tf.convert_to_tensor([occ for _word, occ in self.sorted_voc_with_occ_original], dtype=tf.int64)
        sorted_voc_dataset = tf.data.Dataset.from_tensors(voc)
        sorted_occ_dataset = tf.data.Dataset.from_tensors(occ)

        logger.debug(f"voc shape: {voc.shape}")
        logger.debug(f"occ shape: {occ.shape}")
        tf.data.experimental.save(sorted_voc_dataset, f"{save_path_prefix}{self.extracted_dataset_name}-voc")
        tf.data.experimental.save(sorted_occ_dataset, f"{save_path_prefix}{self.extracted_dataset_name}-occ")
        logger.info(f"Sorted voc dataset saved at: {save_path_prefix}{self.extracted_dataset_name}-voc")
        logger.info(f"Sorted occ dataset saved at: {save_path_prefix}{self.extracted_dataset_name}-occ")
        print(f"Sorted voc serialized and saved at: {save_path_prefix}{self.extracted_dataset_name}-voc")
        print(f"Sorted occ serialized and saved at: {save_path_prefix}{self.extracted_dataset_name}-occ")

    @staticmethod
    def create_connection(path_prefix: str = f"~/email_datasets/", db_file: str = f"emails.db"):
        """ create a database connection to a SQLite database """
        conn = None
        try:
            conn = sqlite3.connect(f"{path_prefix}{db_file}")
            print(f"SQLite 3 version: {sqlite3.version}")
            return conn
        except Error as e:
            print(e)

    def create_tables(self, db_conn):
        if db_conn is None:
            raise ValueError("Connection not instantiated")

        db_conn.execute(self.create_emails_table)
        db_conn.execute(self.create_keywords_table)
        db_conn.execute(self.create_keyword_email_occurrence_table)
        db_conn.commit()

    def insert_data_into_db(self):
        """
        These statements aren't safe, since no prepared statements are used.
        This is by design because it doesn't seem necessary :)
        """
        assert self.sorted_voc_with_occ_original is not None and len(self.sorted_voc_with_occ_original) > 0
        assert self.freq_dict is not None

        db_conn = self.create_connection()

        self.create_tables(db_conn)

        cursor = db_conn.cursor()

        reverse_dict = {}
        for i, (word, occ) in enumerate(self.sorted_voc_with_occ_original):
            reverse_dict[word] = i + 1
            cursor.execute(f"INSERT INTO keywords (keyword) VALUES ('{word}');")

        print("Executed all keywords")

        reverse_emails = {}
        email_id = 1
        for file_name, keywords in self.freq_dict.items():
            reverse_emails[file_name] = email_id
            email_id += 1
            cursor.execute(f"INSERT INTO emails (email) VALUES ('{file_name}');")
            for keyword, occ in keywords.items():
                cursor.execute(f"""INSERT INTO occurrence 
                    (email_id, keyword_id, occurrence)
                    VALUES ({email_id}, {reverse_dict[keyword]}, {occ});""")

        db_conn.commit()

        db_conn.close()

    @staticmethod
    def extract_email_voc(ind, d_frame, one_occ_per_doc=True):
        freq_dict = {}
        glob_freq_list = {}
        for row_tuple in tqdm.tqdm(
                iterable=d_frame.itertuples(),
                desc=f"Extracting corpus vocabulary (Core {ind})",
                total=len(d_frame),
                position=ind,
        ):
            temp_freq_dist = KeywordExtractor.get_voc_from_one_email(
                row_tuple.mail_body, freq=True
            )
            freq_dict[row_tuple.filename] = {}
            for word, freq in temp_freq_dist.items():
                freq_to_add = 1 if one_occ_per_doc else freq
                freq_dict[row_tuple.filename][word] = freq_to_add
                try:
                    glob_freq_list[word] += freq_to_add
                except KeyError:
                    glob_freq_list[word] = freq_to_add
        return freq_dict, glob_freq_list
