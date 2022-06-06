import logging
import multiprocessing
import random

from functools import reduce
from contextlib import contextmanager
from collections import Counter
from typing import List

import colorlog
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from ckws_adapted_score_attack.src.config import DATA_TYPE

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")

MAX_CPUS = 64   # multiprocessing.cpu_count() if multiprocessing.cpu_count() <= 12 else 12


def setup_logger():
    logger.handlers = []  # Reset handlers
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s.%(msecs)03d %(levelname)s]%(reset)s %(module)s: "
            "%(white)s%(message)s",
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


@contextmanager
def poolcontext(*args, **kwargs):
    """Context manager to standardize the parallelized functions.
    """
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


class OccRowComputer:
    """Callable class used to parallelize occurrence matrix computation
    """

    def __init__(self, sorted_voc_with_occ):
        self.voc = [word for word, occ in sorted_voc_with_occ]

    def __call__(self, word_list):
        # Set 1 if word in word_list of a document, otherwise 0
        return [int(voc_word in word_list) for voc_word in self.voc]


class KeywordExtractor:
    """Class to extract the keyword from a corpus/email set
    """

    def __init__(self, corpus_df: pd.DataFrame, voc_size: int = 100, min_freq: int = 1,
                 with_kw_conjunction_size: int = 1):
        glob_freq_dict = {}
        freq_dict = {}
        # Word tokenization

        with poolcontext(processes=MAX_CPUS) as pool:
            # results = [freq_dict_i, global_freq_dict_i for i in range(NUM_CORES)]
            results = pool.starmap(
                self.extract_email_voc, zip(
                    range(MAX_CPUS),
                    np.array_split(corpus_df, MAX_CPUS)
                ),
            )
            freq_dict, glob_freq_dict = reduce(self._merge_results, results)

        # Remove from memory
        del corpus_df

        logger.info(f"Number of unique words (except the stopwords): {len(glob_freq_dict)}")

        # Creation of the vocabulary
        glob_freq_list = nltk.FreqDist(glob_freq_dict)

        # Remove from memory
        del glob_freq_dict

        # Only keep the $voc_size most common words
        glob_freq_list = glob_freq_list.most_common(voc_size) if voc_size else glob_freq_list.most_common()

        # Sort from highest word count to lowest word count
        self.sorted_voc_with_occ_original = sorted(
            [(word, count) for word, count in glob_freq_list if count >= min_freq],
            key=lambda d: d[1],  # d[1] a.k.a. count
            reverse=True,
        )
        logger.info(f"Vocabulary size: {len(self.sorted_voc_with_occ_original)}")

        # Increase sorted vocabulary when using conjunctive keywords (according kw_conjunction_size)
        self.sorted_voc_with_occ, self.keyword_combinations = self.apply_sorted_voc_conjunctive_kws(
            kw_conjunction_size=with_kw_conjunction_size)

        # Remove from memory
        del glob_freq_list

        # Creation of the occurrence matrix
        self.original_occ_array = self.build_occurrence_array(
            sorted_voc_with_occ=self.sorted_voc_with_occ_original,
            freq_dict=freq_dict
        )

        # Column wise increase occ_array according to kw_conjunction_size
        logger.info(f"Element wise matrix multiplication, kw_conjunction_size={with_kw_conjunction_size}")
        # self.occ_array = self.element_wise_occ_array_multiplication(
        #     self.original_occ_array,
        #     kw_conjunction_size=with_kw_conjunction_size
        # )
        if with_kw_conjunction_size > 1:
            self.occ_array = increase_occ_array_element_wise(
                original_occ=self.original_occ_array,
                keyword_combinations=self.keyword_combinations
            )
        else:
            self.occ_array = self.original_occ_array
        logger.info(f"[DONE] Element wise matrix multiplication, kw_conjunction_size={with_kw_conjunction_size}")

        logger.debug(f"self.occ_array: {self.occ_array[:10]}")
        logger.debug(f"Sorted voc: {self.sorted_voc_with_occ}")

        # Check if occurrence array has entries
        if not self.occ_array.shape[0] > 0:
            raise ValueError("occurrence array is empty")

    def get_sorted_voc(self):
        """Returns the sorted vocabulary without the occurrences.

        Returns:
            List -- Word list
        """
        return dict(self.sorted_voc_with_occ).keys()

    def apply_sorted_voc_conjunctive_kws(self, kw_conjunction_size: int = 2):
        num_keywords = len(self.sorted_voc_with_occ_original)

        keyword_combinations = conjunctive_keyword_combinations_indices(
            num_keywords,
            kw_conjunction_size=kw_conjunction_size
        )

        # Transpose for more readable code
        keyword_combinations = keyword_combinations.T

        # New keywords list
        new_applied = []

        # Loop through every possible combination of keywords (according to kw_conjunction_size)
        for i in range(keyword_combinations.shape[0]):
            conjunctive_keyword = tuple()
            conjunctive_count = 1
            for j in range(keyword_combinations.shape[1]):
                conjunctive_keyword += (self.sorted_voc_with_occ_original[keyword_combinations[i, j]][0],)
                conjunctive_count *= self.sorted_voc_with_occ_original[keyword_combinations[i, j]][1]
            new_applied.append((conjunctive_keyword, conjunctive_count))

        return new_applied, keyword_combinations

    def element_wise_occ_array_multiplication(self, occ: tf.Tensor, kw_conjunction_size=2) -> tf.Tensor:
        """Returns the element wise 'multiplication' of matrix columns.
        E.g. (kw_conjunction_size=2):
            w1  w2  w3           w1||w1  w1||w2  w1||w3  w2||w1  w2||w2  w2||w3  w3||w1  w3||w2  w3||w3
        d1  1   1   0   =>  d1   1       1       0       1       1       0       0       0       0
        d2  0   1   1   =>  d2   0       0       0       0       1       1       0       1       1
        """
        conjunctive_occ = occ
        for _ in range(kw_conjunction_size - 1):
            conjunctive_occ = increase_conjunctive_keyword(conjunctive_occ, self.original_occ_array)

        return conjunctive_occ

    @staticmethod
    def _merge_results(res1, res2):
        merge_results2 = Counter(res1[1]) + Counter(res2[1])

        merge_results1 = res1[0].copy()
        merge_results1.update(res2[0])
        return merge_results1, merge_results2

    @staticmethod
    def get_voc_from_one_email(email_text, freq=False):
        stopwords_list = stopwords.words("english")
        stopwords_list.extend(["subject", "cc", "from", "to", "forward"])
        stemmer = PorterStemmer()

        stemmed_word_list = [
            stemmer.stem(word.lower())
            for sentence in sent_tokenize(email_text)
            for word in word_tokenize(sentence)
            if word.lower() not in stopwords_list and word.isalnum()
        ]

        if freq:  # (word, occurrence) sorted list
            return nltk.FreqDist(stemmed_word_list)
        else:  # Word list
            return stemmed_word_list

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
            freq_dict[row_tuple.filename] = []
            for word, freq in temp_freq_dist.items():
                freq_to_add = 1 if one_occ_per_doc else freq
                freq_dict[row_tuple.filename].append(word)
                try:
                    glob_freq_list[word] += freq_to_add
                except KeyError:
                    glob_freq_list[word] = freq_to_add
        return freq_dict, glob_freq_list

    @staticmethod
    def build_occurrence_array(sorted_voc_with_occ: List, freq_dict: dict) -> tf.Tensor:
        occ_list = []
        with poolcontext(processes=MAX_CPUS) as pool:
            for row in tqdm.tqdm(
                    pool.imap_unordered(OccRowComputer(sorted_voc_with_occ), freq_dict.values()),
                    desc=f"Computing the occurrence array",
                    total=len(freq_dict.values()),
            ):
                occ_list.append(row)

        return tf.convert_to_tensor(occ_list, dtype=DATA_TYPE)


def conjunctive_keyword_combinations_indices(num_keywords: int, kw_conjunction_size: int = 2) -> np.ndarray:
    n = num_keywords
    k = kw_conjunction_size
    a = np.ones((k, n - k + 1), dtype=int)
    a[0] = np.arange(n - k + 1)
    for j in range(1, k):
        reps = (n - k + j) - a[j - 1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1 - reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a


def cartesian_product(*arrays):
    la = len(arrays)
    d_type = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=d_type)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def increase_conjunctive_keyword(occ: tf.Tensor, original_occ: tf.Tensor) -> tf.Tensor:
    """
    Helper function for 'element_wise_occ_array_multiplication' function
    """
    new_occ = tf.zeros(shape=(original_occ.shape[0], 0), dtype=DATA_TYPE)

    # For every i-th column in the matrix
    for i in range(original_occ.shape[1]):
        # Hadamard product: [1, 2, 3] * [[2, 4, 3], [4, 8, 9]] = [[2, 8, 9], [4, 16, 27]]
        multiplied = tf.math.multiply(original_occ[:, i], tf.transpose(occ))

        # Concat the result to the starting empty matrix,
        # i.e. A||B, e.g. [[1], [2], [3]] || [[2, 5, 1]] = [[1, 2], [2, 5], [3, 1]]
        # intuitively stick the matrix at the right-end of the matrix -> horizontally expands
        new_occ = tf.concat([new_occ, tf.transpose(multiplied)], axis=1)
    return new_occ


# @ray.remote
def increase_occ_array_element_wise(
        original_occ: tf.Tensor,
        keyword_combinations: tf.Tensor,
        core: int = -1,
) -> tf.Tensor:
    return increase_occ_array_element_wise_single_core(
        original_occ=original_occ, keyword_combinations=keyword_combinations, core=core)


@tf.function
def increase_occ_array_element_wise_per_combination_multi_core(
        original_occ: tf.Tensor,
        keyword_combinations: tf.Tensor,
) -> tf.Tensor:
    column_vectors = tf.TensorArray(
        dtype=DATA_TYPE,
        size=tf.shape(keyword_combinations)[0],
        element_shape=tf.TensorShape(dims=[original_occ.shape[0]]),
        dynamic_size=False,
        name="init_tensorarray_conjunctive_occ_array"
    )

    def tf_while_body(idx, results):
        keyword_vectors = tf.gather(
            original_occ,
            keyword_combinations[idx],
            axis=1,
            name="gather_occ_array_single_core"
        )

        # Multiply the keyword vectors
        multiplied = tf.math.reduce_prod(keyword_vectors, axis=1, name="reduce_prod_keyword_vectors_single_core")

        return tf.add(idx, 1, name="add_idx_add_1_increase_occ_array_while_loop"), \
               results.write(idx, multiplied, name="write_column_vector_into_tensor_array")

    i = tf.constant(0)
    tf_while_cond = lambda loop_idx, x: tf.less(loop_idx, tf.shape(keyword_combinations)[0])

    idx, result = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[i, column_vectors],
        # shape_invariants=[i.get_shape(), tf.TensorShape([tf.shape(keyword_combinations)[0], None])],
        parallel_iterations=MAX_CPUS,
        name="while_loop_conjunctive_keyword_document_array"
    )

    return result.stack(name="stack_conjunctive_keyword_vectors")


@tf.function
def increase_occ_array_element_wise_single_core(
        original_occ: tf.Tensor,
        keyword_combinations: tf.Tensor,
        core: int = -1,
) -> tf.Tensor:
    # Empty list of column vectors
    logger.debug(f"single core, keyword_combinations: {keyword_combinations}")
    column_vectors = tf.TensorArray(
        dtype=DATA_TYPE,
        size=tf.shape(keyword_combinations)[0],
        dynamic_size=False,
        name="init_tensorarray_conjunctive_occ_array"
    )

    # For every keyword combination rebuild the keyword-document matrix
    logger.info(f"Number of keyword combinations: {keyword_combinations.shape[0]}")
    idx = tf.constant(0, name="initialize_idx_0_constant")
    for keyword_combination in keyword_combinations:
        # Create matrix of a keyword combination
        keyword_vectors = tf.gather(original_occ, keyword_combination, axis=1, name="gather_occ_array_single_core")

        # Multiply the keyword vectors
        multiplied = tf.math.reduce_prod(keyword_vectors, axis=1, name="reduce_prod_keyword_vectors_single_core")

        # Append new keyword vector to columns
        column_vectors = column_vectors.write(idx, multiplied, name="write_column_vector_into_tensor_array")

        idx = tf.add(idx, 1, name="increase_idx_to_next_keyword_combination")

    # Concat all the columns to the starting empty matrix,
    # i.e. A||B, e.g. [[1], [2], [3]] || [[2, 5, 1]] = [[1, 2], [2, 5], [3, 1]]
    # intuitively stick the matrix at the right-end of the matrix -> 'horizontally expands'
    # new_occ = tf.concat([new_occ, tf.transpose()], axis=1)

    return tf.transpose(column_vectors.stack(name="stack_conjunctive_keyword_vectors"), name="transpose_new_occ_array")


# https://stackoverflow.com/questions/53541803/rows-or-elements-selection-on-sparse-tensor
def sparse_select_indices(sp_input, indices, axis=0):
    # Only necessary if indices may have non-unique elements
    indices, _ = tf.unique(indices)
    n_indices = tf.size(indices)
    # Only necessary if indices may not be sorted
    indices, _ = tf.math.top_k(indices, n_indices)
    indices = tf.reverse(indices, [0])
    # Get indices for the axis
    idx = sp_input.indices[:, axis]
    # Find where indices match the selection
    eq = tf.equal(tf.expand_dims(idx, 1), tf.cast(indices, tf.int64))
    # Mask for selected values
    sel = tf.reduce_any(eq, axis=1)
    # Selected values
    values_new = tf.boolean_mask(sp_input.values, sel, axis=0)
    # New index value for selected elements
    n_indices = tf.cast(n_indices, tf.int64)
    idx_new = tf.reduce_sum(tf.cast(eq, tf.int64) * tf.range(n_indices), axis=1)
    idx_new = tf.boolean_mask(idx_new, sel, axis=0)
    # New full indices tensor
    indices_new = tf.boolean_mask(sp_input.indices, sel, axis=0)
    indices_new = tf.concat([indices_new[:, :axis],
                             tf.expand_dims(idx_new, 1),
                             indices_new[:, axis + 1:]], axis=1)
    # New shape
    shape_new = tf.concat([sp_input.dense_shape[:axis],
                           [n_indices],
                           sp_input.dense_shape[axis + 1:]], axis=0)
    return tf.SparseTensor(indices_new, values_new, shape_new)


def generate_known_queries(similar_wordlist, stored_wordlist, nb_queries):
    """Extract random keyword which are present in the similar document set
    and in the server. So the pairs (similar_keyword, trapdoor_keyword) will
    be considered as known queries. Since the trapdoor words are not hashed
    the tuples will be like ("word","word"). We could only return the keywords
    but this tuple represents well what an attacker would have, i.e. tuple linking
    one keyword to a trapdoor they has seen.

    NB: the length of the server wordlist is the number of possible queries

    Arguments:
        similar_wordlist {List[str]} -- List of the keywords of the similar vocabulary
        trapdoor_wordlist {List[str]} -- List of the keywords of the server vocabulary
        nb_queries {int} -- Number of queries wanted

    Returns:
        dict[str,str] -- dictionary containing known queries
    """
    candidates = set(similar_wordlist).intersection(stored_wordlist)
    return {word: word for word in random.sample(candidates, nb_queries)}
