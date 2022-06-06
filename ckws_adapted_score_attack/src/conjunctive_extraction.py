import hashlib
import multiprocessing
import random
from typing import Tuple, Dict, List

import tensorflow as tf
import numpy as np

import colorlog

from ckws_adapted_score_attack.src.common import conjunctive_keyword_combinations_indices, \
    increase_occ_array_element_wise, increase_occ_array_element_wise_single_core, \
    increase_occ_array_element_wise_per_combination_multi_core
from ckws_adapted_score_attack.src.config import DATA_TYPE
from ckws_adapted_score_attack.src.trapdoor_matchmaker import _generate_splits_by_number_of_cpu

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")

MAX_CPUS = 50  # multiprocessing.cpu_count() if multiprocessing.cpu_count() <= 128 else 128


class ConjunctiveExtractor:
    """
    Class to extract occurrence array and global keyword + occurrence from preprocessed dataset
    Also able to generate occurrence array for an artificial keyword size


    Artificial keyword combinations is optional, but can be used to save time when running a lot of experiments.
        NB: The keyword combinations won't change if you run many examples on just different dataset splits.
    """

    def __init__(
            self,
            occurrence_array: tf.SparseTensor,
            keyword_voc: tf.Tensor,
            keyword_occ: tf.Tensor,
            voc_size: int = 100,
            kw_conjunction_size: int = 1,
            min_freq: int = 1,
            precalculated_artificial_keyword_combinations_indices: np.ndarray = None,
            multi_core: bool = True,
    ):
        assert voc_size <= occurrence_array.shape[1], "Vocabulary size should be less or equal than total keywords"

        self.kw_conjunction_size = kw_conjunction_size
        self.voc_size = voc_size
        self.min_freq = min_freq

        # Calculate global occurrence per keyword
        column_sums = tf.sparse.reduce_sum(sp_input=occurrence_array, axis=0)

        logger.debug(f"len column sums: {column_sums.shape}")

        # Sort the global keywords and pick 'voc_size' most common
        self.original_keyword_indices = tf.cast(tf.argsort(column_sums, direction='DESCENDING'), dtype=tf.int64)
        self.original_keyword_indices = self.original_keyword_indices[:self.voc_size]

        # Store keywords tensor{'string'}
        self.keywords = tf.gather(keyword_voc, self.original_keyword_indices)

        # Column sums, since occurrence will be different each experiment
        self.occurrences = tf.gather(column_sums, self.original_keyword_indices)

        logger.info(f"Vocabulary size: {self.keywords.shape[0]}")

        # Reduce occurrence array size to match with keyword 'voc_size'
        # Ugly code since tf.gather does not work with sparse tensors
        logger.debug(f"Gather occurrence matrix columns for vocabulary")
        _occ_array = sparse_gather(occurrence_array, self.original_keyword_indices, axis=1)
        _occ_array = tf.sparse.reorder(_occ_array)
        self.original_occ_array = tf.sparse.to_dense(_occ_array)
        # logger.debug(f"Original occ array: {self.original_occ_array[:10]}")

        if self.kw_conjunction_size > 1:
            # Make keyword combinations based on kw_conjunction_size
            logger.debug(f"Generate keyword combinations and sorted vocabulary")

            if precalculated_artificial_keyword_combinations_indices is None:
                self.keyword_combinations = conjunctive_keyword_combinations_indices(
                    num_keywords=self.voc_size,
                    kw_conjunction_size=self.kw_conjunction_size
                ).T
            else:
                self.keyword_combinations = precalculated_artificial_keyword_combinations_indices.T

            # Get original keyword combinations indices and sort on original occurrence
            self.voc_indices = tf.sort(tf.gather(self.original_keyword_indices, self.keyword_combinations))

            # Get keywords and concatenate
            self.sorted_voc = tf.strings.reduce_join(tf.gather(keyword_voc, self.voc_indices), separator="|", axis=-1)

            # Rebuild occurrence array based on keyword combinations
            logger.info(f"Element wise matrix multiplication, kw_conjunction_size={self.kw_conjunction_size}")

            if multi_core:
                # splits = tf.convert_to_tensor(_generate_splits_by_number_of_cpu(self.keyword_combinations.shape[0]))
                logger.debug(f"self.keyword_combinations.shape: {self.keyword_combinations.shape}")
                # np.array_split(self.keyword_combinations, MAX_CPUS)

                # self.occ_array = increase_occ_array_element_wise_multiprocessing(
                #     original_occ=self.original_occ_array,
                #     keyword_combinations=tf.convert_to_tensor(self.keyword_combinations, name="kw_combos_to_tensor"),
                #     splits=splits,
                # )

                self.occ_array = increase_occ_array_element_wise_per_combination_multi_core(
                    original_occ=self.original_occ_array,
                    keyword_combinations=self.keyword_combinations,
                )
            else:
                self.occ_array = increase_occ_array_element_wise_single_core(
                    original_occ=self.original_occ_array,
                    keyword_combinations=self.keyword_combinations
                )
        else:
            self.voc_indices = self.original_keyword_indices
            self.sorted_voc = self.keywords
            self.occ_array = self.original_occ_array

        logger.debug(f"Occ array: {self.occ_array.shape}")

    def get_sorted_voc(self) -> tf.Tensor:
        """Returns the sorted vocabulary without the occurrences.

        Returns:
            tf.Tensor<tf.string> -- Vocabulary (concatenated) keywords list
        """
        return self.sorted_voc


@tf.function
def increase_occ_array_element_wise_multiprocessing(
        original_occ: tf.Tensor,
        keyword_combinations: tf.Tensor,
        splits: tf.Tensor,
) -> tf.Tensor:
    init_results = tf.zeros(shape=(original_occ.shape[0], 0), dtype=DATA_TYPE, name="init_results_tf_zeros")

    def tf_while_body(idx, results):
        common_split_size = splits[0]
        current_split_size = splits[idx]

        current_keyword_comb_split = tf.slice(keyword_combinations,
                                              begin=[idx * common_split_size, 0],
                                              size=[current_split_size, keyword_combinations.shape[1]],
                                              name="while_keyword_combination_slice")

        sub_result = increase_occ_array_element_wise(
            original_occ=original_occ,
            keyword_combinations=current_keyword_comb_split
        )

        idx = tf.add(idx, 1, name="add_idx_add_1_increase_occ_array_while_loop")

        return idx, tf.concat([results, sub_result], axis=1, name="concat_artificial_keyword_document_array")

    i = tf.constant(0)
    tf_while_cond = lambda loop_idx, x: tf.less(loop_idx, splits.shape[0])

    idx, result = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[i, init_results],
        shape_invariants=[i.get_shape(), tf.TensorShape([init_results.shape[0], None])],
        parallel_iterations=MAX_CPUS,
        name="while_loop_artificial_keyword_document_array"
    )

    return result


def sparse_gather(sparse_tensor: tf.SparseTensor, selected_indices: tf.Tensor, axis: int = 0) -> tf.SparseTensor:
    """
    indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
    values:  [ value1,                                 , ..., valuen]
    """
    indices = sparse_tensor.indices
    mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
    to_select = tf.where(mask)[:, 1]
    new_indices = tf.gather(indices, to_select, axis=0, name="gather_indices_for_sparse_gather")
    new_values = tf.gather(sparse_tensor.values, to_select, axis=0, name="gather_values_for_sparse_gather")

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=selected_indices,
            values=tf.range(selected_indices.shape[0], dtype=tf.int64),
        ),
        default_value=tf.constant(-1, dtype=tf.int64),
        name="class_weight"
    )

    # Mapping column indices from old vocabulary to the new/reduced vocabulary
    mapped_indices = table.lookup(new_indices[:, 1])

    # Stack s.t. the sparse tensor has the right indices
    new_indices = tf.stack([new_indices[:, 0], mapped_indices], axis=1)

    new_shape = [sparse_tensor.dense_shape[0], len(selected_indices)] if axis == 1 else [len(selected_indices),
                                                                                         sparse_tensor.dense_shape[1]]
    tensor = tf.SparseTensor(indices=new_indices, values=new_values, dense_shape=new_shape)

    # Reorder since indices may not necessarily be ordered
    tensor = tf.sparse.reorder(tensor, name="reorder_sparse_gather")

    return tensor


def generate_known_queries_indices(similar_wordlist: tf.Tensor, stored_wordlist: tf.Tensor, nb_queries: int) \
        -> tf.Tensor:
    """Extract random keyword which are present in the similar document set
    and in the server. So the pairs (similar_keyword, trapdoor_keyword) will
    be considered as known queries. Since the trapdoor words are not hashed
    the tuples will be like ("word","word"). We could only return the keywords
    but this tuple represents well what an attacker would have, i.e. tuple linking
    one keyword to a trapdoor they has seen.

    NB: the length of the server wordlist is the number of possible queries

    Arguments:
        similar_wordlist {tf.Tensor<int32>} -- Similar vocabulary global keyword indices
        stored_wordlist {tf.Tensor<int32>} -- Server vocabulary global (trapdoor/)keyword indices
        nb_queries {int} -- Number of queries wanted

    Returns:
        tf.Tensor<dtype=tf.int32, shape=(nb_queries, kw_conjunction_size)> -- Tensor containing known queries
    """
    find_match = tf.reduce_sum(tf.abs(tf.expand_dims(stored_wordlist, 0) - tf.expand_dims(similar_wordlist, 1)), 2)

    indices = tf.transpose(tf.where(tf.equal(find_match, tf.zeros_like(find_match))))[0]

    intersection = tf.gather(similar_wordlist, indices)

    return tf.random.shuffle(intersection)[:nb_queries]
    # return {word: word for word in random.sample(candidates, nb_queries)}


def generate_known_queries(similar_wordlist: tf.Tensor, stored_wordlist: tf.Tensor, nb_queries: int) -> Dict[str, str]:
    """Extract random keyword which are present in the similar document set
    and in the server. So the pairs (similar_keyword, trapdoor_keyword) will
    be considered as known queries. Since the trapdoor words are not hashed
    the tuples will be like ("word","word"). We could only return the keywords
    but this tuple represents well what an attacker would have, i.e. tuple linking
    one keyword to a trapdoor they has seen.

    NB: the length of the server wordlist is the number of possible queries

    Arguments:
        similar_wordlist {tf.Tensor<tf.string>} -- 1D tensor of the keywords of the similar vocabulary
        trapdoor_wordlist {tf.Tensor<tf.string>} -- 1D tensor of the keywords of the server vocabulary
        nb_queries {int} -- Number of queries wanted

    Returns:
        dict[str,str] -- dictionary containing known queries
    """
    candidates = set(similar_wordlist).intersection(stored_wordlist)
    return {word: word for word in random.sample(candidates, nb_queries)}


def generate_trapdoors(query_voc: tf.Tensor, known_queries: Dict[str, str]) \
        -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    # Trapdoor token => convey no information about the corresponding keyword
    temp_voc = []
    temp_known = {}
    eval_dict = {}  # Keys: Trapdoor tokens; Values: Keywords, 'the answers'
    for keyword in query_voc:
        # We replace each keyword of the trapdoor dictionary by its hash
        # So the matchmaker truly ignores the keywords behind the trapdoors.
        fake_trapdoor = hashlib.sha1(keyword).hexdigest()
        temp_voc.append(fake_trapdoor)

        if known_queries.get(keyword):
            temp_known[fake_trapdoor] = keyword

        eval_dict[fake_trapdoor] = keyword
    query_voc = temp_voc
    known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

    return query_voc, known_queries, eval_dict


def _contains_row(hay_stack: tf.Tensor, row: tf.Tensor) -> bool:
    assert hay_stack.shape[1] == row.shape[1], "Must have equal shape on axis 1"

    prediction = tf.reduce_prod(tf.where(tf.equal(hay_stack, row), [1] * len(row), 0), axis=1)
    contains = tf.reduce_max(prediction)

    return contains == 1
