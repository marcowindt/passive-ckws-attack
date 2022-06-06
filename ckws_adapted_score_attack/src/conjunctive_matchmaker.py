import gc
import math
import multiprocessing

from functools import partial
from typing import Dict, List, Optional

import colorlog
import numpy as np
import tensorflow as tf
import time

import tqdm

from ckws_adapted_score_attack.src.config import DATA_TYPE, MATMUL_BATCHES_KW, MATMUL_BATCHES_TD, PROFILE_MATMUL
from ckws_adapted_score_attack.src.trapdoor_matchmaker import predict_trapdoors, predict_trapdoors_k, pred_td_list

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")


class ConjunctiveScoreAttack:
    def __init__(
        self,
        keyword_occ_array: tf.Tensor,
        keyword_sorted_voc: tf.Tensor,
        trapdoor_occ_array: tf.Tensor,
        trapdoor_sorted_voc: Optional[List[str]],
        known_queries: Dict[str, str],
        norm_ord=2,  # L2 (Euclidean norm),
        **kwargs,
    ):
        """Initialization of the matchmaker

        Arguments:
            keyword_occ_array {np.array} -- Keyword occurrence (row: similar documents; columns: keywords)
            trapdoor_occ_array {np.array} -- Trapdoor occurrence (row: stored documents; columns: trapdoors)
                                            the documents are unknown (just the identifier has
                                            been seen by the attacker)
            keyword_sorted_voc {List[str]} -- Keyword vocabulary extracted from similar documents.
            known_queries {Dict[str, str]} -- Queries known by the attacker
            trapdoor_sorted_voc {Optional[List[str]]} -- The trapdoor voc can be a sorted list of hashes
                                                            to hide the underlying keywords.

        Keyword Arguments:
            norm_ord {int} -- Order of the norm used by the matchmaker (default: {2})
        """
        self.set_norm_ord(norm_ord=norm_ord)
        self.set_tf_norm_ord(norm_ord='euclidean')

        if not known_queries:
            raise ValueError("Known queries are mandatory.")
        if len(known_queries.values()) != len(set(known_queries.values())):
            raise ValueError("Several trapdoors are linked to the same keyword.")

        self.known_queries = known_queries.copy()  # Keys: trapdoor, Values: keywords{tf.Tensor}

        self.number_similar_docs = keyword_occ_array.shape[1]

        # NB: kw=KeyWord; td=TrapDoor
        self.kw_voc_info = {
            word: {"vector_ind": ind}   # , "word_occ": tf.reduce_sum(keyword_occ_array[:, ind])}
            for ind, word in enumerate(keyword_sorted_voc)
        }

        logger.debug(f"len keyword_sorted_voc: {keyword_sorted_voc.shape}")
        logger.debug(f"len        kw_voc_info: {len(self.kw_voc_info)}")

        self.td_voc_info = {
            word: {"vector_ind": ind}   # , "word_occ": tf.reduce_sum(trapdoor_occ_array[:, ind])}
            for ind, word in enumerate(trapdoor_sorted_voc)
        }

        logger.debug(f"len trapdoor_sorted_voc: {len(trapdoor_sorted_voc)}")
        logger.debug(f"len         td_voc_info: {len(self.td_voc_info)}")

        keyword_occ_sums = tf.reduce_sum(keyword_occ_array, axis=1)
        trapdoor_occ_sums = tf.reduce_sum(trapdoor_occ_array, axis=1)

        self.number_real_docs = self._estimate_nb_real_docs(kws_occ=keyword_occ_sums, tds_occ=trapdoor_occ_sums)

        del keyword_occ_sums
        del trapdoor_occ_sums

        logger.debug(f"Estimated number of real docs: {self.number_real_docs}")
        logger.debug(f"Number of similar docs: {self.number_similar_docs}")

        self.keyword_occ_array = keyword_occ_array
        self.trapdoor_occ_array = trapdoor_occ_array

        self._compute_coocc_matrices()

    def _generate_splits_by_number(self, axis_length: int, x: int):
        split_size = int(math.ceil(axis_length / x))
        too_much = x * split_size - axis_length
        splits = [split_size for _ in range(x)]

        while too_much / split_size > 1:
            too_much -= split_size
            splits.pop()

        splits[-1] -= too_much

        if splits[-1] == 0:
            splits.pop()

        return splits

    def _compute_coocc_matrices(self):
        # TODO: improve with sparse Tensors -- Not possible, since there is no efficient matmul function for Sparse
        logger.info(f"Computing keyword co-occurrence matrices {self.keyword_occ_array.shape}, "
                    f"{self.keyword_occ_array.dtype}")
        logger.info(f"Computing trapdoor co-occurrence matrices {self.trapdoor_occ_array.shape}, "
                    f"{self.trapdoor_occ_array.dtype}")

        # ARTIFICIAL_KEYWORD CO-OCCURRENCE
        if PROFILE_MATMUL:
            tf.profiler.experimental.start('profiler/log')
        # Dot product keyword occurrence matrix: with only looking at known queries beforehand to save memory
        known_queries_indices = self._get_kw_known_queries_indices_enumerated()

        batches = MATMUL_BATCHES_KW

        kw_coocc = tf.zeros(shape=(self.keyword_occ_array.shape[0], len(known_queries_indices)), dtype=DATA_TYPE)

        kw_splits = self._generate_splits_by_number(axis_length=self.keyword_occ_array.shape[1], x=batches)

        with tf.device("/device:CPU:0"):
            keyword_occ_array_b = tf.gather(
                self.keyword_occ_array,
                known_queries_indices[:, 0],
                axis=0,
                name="gather_batch_kw_occ_array"
            )

        for batch in tqdm.tqdm(iterable=range(batches), desc="matmul batch kw coocc", total=batches):
            with tf.device("/device:CPU:0"):
                common_split_size = kw_splits[0]
                current_split_size = kw_splits[batch]

                batch_keyword_occ_array_a = tf.slice(
                    self.keyword_occ_array,
                    begin=[0, batch * common_split_size],
                    size=(self.keyword_occ_array.shape[0], current_split_size)
                )
                logger.debug(f"batch_keyword_occ_array_a.shape: {batch_keyword_occ_array_a.shape}")
                batch_keyword_occ_array_b = tf.slice(
                    keyword_occ_array_b,
                    begin=[0, batch * common_split_size],
                    size=(keyword_occ_array_b.shape[0], current_split_size)
                )
                logger.debug(f"batch_keyword_occ_array_b.shape: {batch_keyword_occ_array_b.shape}")

            with tf.device("/device:GPU:0"):
                batch_keyword_occ_array_a = tf.identity(batch_keyword_occ_array_a)
                batch_keyword_occ_array_b = tf.identity(batch_keyword_occ_array_b)

                batch_kw_coocc = tf.linalg.matmul(
                    a=batch_keyword_occ_array_a,
                    b=batch_keyword_occ_array_b,
                    transpose_b=True,
                    name="batch_matmul_kw_coocc",
                )

            with tf.device("/device:CPU:0"):
                batch_kw_coocc = tf.identity(batch_kw_coocc)

                kw_coocc = tf.add(kw_coocc, batch_kw_coocc)

        with tf.device("/device:CPU:0"):
            kw_coocc = kw_coocc / self.number_similar_docs

            diagonals = tf.sparse.SparseTensor(
                indices=known_queries_indices,
                values=tf.ones(shape=(len(known_queries_indices),), dtype=DATA_TYPE, name="sparse_known_queries_ones"),
                dense_shape=kw_coocc.shape,
            )

            diagonals = tf.sparse.reorder(diagonals, name="sparse_reorder_kws_diagonals")

            self.kw_reduced_coocc = tf.subtract(
                kw_coocc,
                tf.sparse.to_dense(kw_coocc * diagonals, name="sparse_to_dense_kw_coocc_diagonals"),
                name="subtract_diagonals_from_kw_coocc"
            )

        # logger.info(f"Computed keyword co-occurrence matrix (1/2) {self.kw_coocc.shape}")
        if PROFILE_MATMUL:
            tf.profiler.experimental.stop()

        # TRAPDOOR CO-OCCURRENCE
        if PROFILE_MATMUL:
            tf.profiler.experimental.start('profiler/log')

        trapdoor_queries_indices = self._get_td_known_queries_indices_enumerated()
        batches = MATMUL_BATCHES_TD
        td_coocc = tf.zeros(shape=(self.trapdoor_occ_array.shape[0], len(trapdoor_queries_indices)), dtype=DATA_TYPE)

        td_splits = self._generate_splits_by_number(axis_length=self.trapdoor_occ_array.shape[1], x=batches)

        with tf.device("/device:CPU:0"):
            trapdoor_occ_array_b = tf.gather(
                self.trapdoor_occ_array,
                trapdoor_queries_indices[:, 0],
                axis=0,
                name="gather_batch_kw_occ_array"
            )

        for batch in tqdm.tqdm(iterable=range(batches), desc="matmul batch td coocc", total=batches):
            with tf.device("/device:CPU:0"):
                common_split_size = td_splits[0]
                current_split_size = td_splits[batch]

                batch_trapdoor_occ_array_a = tf.slice(
                    self.trapdoor_occ_array,
                    begin=[0, batch * common_split_size],
                    size=(self.trapdoor_occ_array.shape[0], current_split_size)
                )
                logger.debug(f"batch_trapdoor_occ_array_a.shape: {batch_trapdoor_occ_array_a.shape}")
                batch_trapdoor_occ_array_b = tf.slice(
                    trapdoor_occ_array_b,
                    begin=[0, batch * common_split_size],
                    size=(trapdoor_occ_array_b.shape[0], current_split_size)
                )
                logger.debug(f"batch_trapdoor_occ_array_b.shape: {batch_trapdoor_occ_array_b.shape}")

            with tf.device("/device:GPU:0"):
                batch_trapdoor_occ_array_a = tf.identity(batch_trapdoor_occ_array_a)
                batch_trapdoor_occ_array_b = tf.identity(batch_trapdoor_occ_array_b)

                batch_td_coocc = tf.linalg.matmul(
                    a=batch_trapdoor_occ_array_a,
                    b=batch_trapdoor_occ_array_b,
                    transpose_b=True,
                    name="batch_matmul_kw_coocc",
                )

            with tf.device("/device:CPU:0"):
                batch_td_coocc = tf.identity(batch_td_coocc)

                td_coocc = tf.add(td_coocc, batch_td_coocc)

        with tf.device("/device:CPU:0"):
            td_coocc = td_coocc / self.number_real_docs

            diagonals = tf.sparse.SparseTensor(
                indices=trapdoor_queries_indices,
                values=tf.ones(shape=(len(trapdoor_queries_indices),), dtype=DATA_TYPE, name="sparse_td_queries_ones"),
                dense_shape=td_coocc.shape
            )

            diagonals = tf.sparse.reorder(diagonals, name="sparse_reorder_kws_diagonals")

            self.td_reduced_coocc = tf.subtract(
                td_coocc,
                tf.sparse.to_dense(td_coocc * diagonals, name="sparse_to_dense_kw_coocc_diagonals"),
                name="subtract_diagonals_from_kw_coocc"
            )

        logger.info(f"kw_reduced_coocc: {self.kw_reduced_coocc.shape}, td_reduced_coocc: {self.td_reduced_coocc.shape}")

        # with tf.device("/device:GPU:0"):
        #     self.kw_reduced_coocc = tf.identity(self.kw_reduced_coocc)
        #     self.td_reduced_coocc = tf.identity(self.td_reduced_coocc)

        # logger.info(f"Computed trapdoor co-occurrence matrix (2/2) {self.td_coocc.shape}")
        if PROFILE_MATMUL:
            tf.profiler.experimental.stop()

    def _get_kw_known_queries_indices(self):
        """
        Get the known queries vocabulary indices for the keyword occurrence array
        """
        return [
            self.kw_voc_info[kw]["vector_ind"] for kw in self.known_queries.values()
        ]

    def _get_kw_known_queries_indices_enumerated(self) -> np.ndarray:
        """
        Get the known queries vocabulary indices for the keyword occurrence array with enumeration
        """
        return np.array([
            [self.kw_voc_info[kw]["vector_ind"], i] for i, kw in enumerate(self.known_queries.values())
        ])

    def _get_td_known_queries_indices(self):
        """
        Get the known queries vocabulary indices for the trapdoor occurrence array
        """
        return [
            self.td_voc_info[td]["vector_ind"] for td in self.known_queries.keys()
        ]

    def _get_td_known_queries_indices_enumerated(self) -> np.ndarray:
        """
        Get the known queries vocabulary indices for the trapdoor occurrence array with enumeration
        """
        return np.array([
            [self.td_voc_info[td]["vector_ind"], i] for i, td in enumerate(self.known_queries.keys())
        ])

    def _refresh_reduced_coocc(self):
        """Refresh the co-occurrence matrix based on the known queries.

        Since in the co-occurrence matrix calculation we already take the known queries into account
        we simply need to re-do it :)
        """
        logger.info("Refresh co-occurrence matrix based on known queries")

        tf.keras.backend.clear_session()
        gc.collect()

        self._compute_coocc_matrices()

        # with tf.device("/device:CPU:0"):
        #     temp_kw_reduced_coocc = tf.gather(self.kw_coocc, self._get_kw_known_queries_indices(), axis=1)
        #     temp_td_reduced_coocc = tf.gather(self.td_coocc, self._get_td_known_queries_indices(), axis=1)
        #
        # with tf.device("/device:GPU:0"):
        #     self.kw_reduced_coocc = tf.identity(temp_kw_reduced_coocc)
        #     self.td_reduced_coocc = tf.identity(temp_td_reduced_coocc)
        #
        # del temp_kw_reduced_coocc
        # del temp_td_reduced_coocc

        logger.info(f"Refreshed co-occurrence matrix based on known queries")

        try:
            mem_gpu = tf.config.experimental.get_memory_info('GPU:0')
            mem_cpu = tf.config.experimental.get_memory_info('CPU:0')
            logger.debug(f"mem_gpu: {mem_gpu}")
            logger.debug(f"mem_cpu: {mem_cpu}")
        except:
            logger.debug(f"could not get_memory_info")

    def _estimate_nb_real_docs(self, kws_occ: tf.Tensor, tds_occ: tf.Tensor):
        """Estimates the number of documents stored.
        """
        nb_doc_ratio_estimator = np.mean(
            [
                tds_occ[self.td_voc_info[td]["vector_ind"]] / kws_occ[self.kw_voc_info[kw]["vector_ind"]]
                for td, kw in self.known_queries.items()
            ]
        )

        return self.number_similar_docs * nb_doc_ratio_estimator

    def set_norm_ord(self, norm_ord):
        """Set the order of the norm used to compute the scores.

        Arguments:
            norm_ord {int} -- norm order
        """
        self._norm = partial(np.linalg.norm, ord=norm_ord)

    def set_tf_norm_ord(self, norm_ord: str = 'euclidean'):
        self._tf_norm_ord = norm_ord

    def _sub_pred_tf_optimized(
        self,
        _ind,
        td_list,
        k=0,
        cluster_max_size=0,
        cluster_min_sensitivity=0,
        include_score=False,
        include_cluster_sep=False,
        multi_core=False,
    ):
        """
        Sub-function used to parallelize the prediction.
        Specify either k for k-highest score mode or cluster_max_size for cluster mode

        Returns:
            List[Tuple[trapdoor,List[keyword]]] -- a list of tuples (trapdoor, predictions)
        """
        if bool(k) == bool(cluster_max_size) or (k and include_cluster_sep):
            raise ValueError("You have to choose either cluster mode or k-best mode")

        # if not multi_core:
        prediction = []
        for trapdoor in tqdm.tqdm(
                iterable=td_list,
                total=len(td_list),
                desc=f"Prediction of trapdoors ({_ind})"):
            predicted_result = _sub_pred_tf_optimized_one_trapdoor_single_core(
                _ind,
                td_voc_info=self.td_voc_info,
                td_reduced_coocc=self.td_reduced_coocc,
                kw_reduced_coocc=self.kw_reduced_coocc,
                trapdoor=trapdoor,
                k=k,
                cluster_max_size=cluster_max_size,
                cluster_min_sensitivity=cluster_min_sensitivity,
                include_score=include_score,
                include_cluster_sep=include_cluster_sep
            )

            prediction.append(predicted_result)

        return prediction

    def tf_predict(self, trapdoor_list, k=None):
        """
        Returns a prediction for each trapdoor in the list. No refinement. No clustering.
        Arguments:
            trapdoor_list {List[str]} -- List of the trapdoors to match with a keyword

        Keyword Arguments:
            k {int} -- the number of highest score keyword wanted for each trapdoor (default: {None})

        Returns:
            dict[trapdoor, predictions] -- dictionary with a list of k predictions for each trapdoor
        """
        if k is None:
            k = len(self.kw_voc_info)

        local_kws_list = list(self.kw_voc_info.keys())

        logger.info("Evaluating every possible keyword-trapdoor pair")

        logger.debug(f"len local_kws_list: {len(local_kws_list)}")
        logger.debug(f"len  trapdoor_list: {len(trapdoor_list)}")

        results = predict_trapdoors_k(
            td_voc_info=self.td_voc_info,
            td_list=trapdoor_list,
            kw_reduced_coocc=self.kw_reduced_coocc,
            td_reduced_coocc=self.td_reduced_coocc,
            k=k,
        )

        # logger.debug(f"predict_k, results: {results}")

        prediction = [
            (td, [local_kws_list[ind] for ind in cluster_kw_indices])
            for td, cluster_kw_indices in results
        ]

        prediction = dict(prediction)

        return prediction

    def tf_predict_with_refinement(self, trapdoor_list, cluster_max_size=10, ref_speed=0):
        """Returns a cluster of predictions for each trapdoor using refinement.

            Arguments:
                trapdoor_list {List[str]} -- List of the trapdoors to match with a keyword

            Keyword Arguments:
                cluster_max_size {int} -- Maximum size of the clusters (default: {10})
                ref_speed {int} -- Refinement speed, i.e. number of queries "learnt" after each iteration (default: {0})

            Returns:
                dict[trapdoor, predictions] -- dictionary with a list of k predictions for each trapdoor
            """
        if ref_speed < 1:
            # Default refinement speed: 5% of the total number of trapdoors
            ref_speed = int(0.05 * len(self.td_voc_info))
        old_known = self.known_queries.copy()

        local_kws_list = list(self.kw_voc_info.keys())

        local_td_list = list(trapdoor_list)

        final_results = []
        while True:
            prev_td_nb = len(local_td_list)
            local_td_list = [
                td for td in local_td_list if td not in self.known_queries.keys()
            ]  # Removes the known trapdoors

            if len(local_td_list) == 0:
                # Rare occasion that there are not tds left to predict
                break

            logger.debug(f"Start prediction: len td_list {len(local_td_list)}")
            start_time = time.time()

            results = pred_td_list(
                td_list=local_td_list,
                td_voc_info=self.td_voc_info,
                kw_reduced_coocc=self.kw_reduced_coocc,
                td_reduced_coocc=self.td_reduced_coocc,
                cluster_max_size=cluster_max_size,
                include_score=False,
                include_cluster_sep=True,
            )

            duration = time.time() - start_time
            logger.debug(f"Done prediction, took: {duration}")

            # Extract the best predictions (must be single-point clusters)
            single_point_results = [tup for tup in results if tup[1].shape[0] == 1]
            single_point_results.sort(key=lambda tup: tup[2])

            if len(single_point_results) < ref_speed:
                # Not enough single-point predictions
                final_results = [
                    (td, [local_kws_list[ind] for ind in cluster_kw_indices])
                    for td, cluster_kw_indices, _sep in results]
                break

            # Add the pseudo-known queries.
            new_known = {
                td: local_kws_list[list_candidates_indices[list_candidates_indices.shape[0] - 1]]
                for td, list_candidates_indices, _sep in single_point_results[-ref_speed:]
            }

            # logger.debug(f"Update known queries")
            self.known_queries.update(new_known)
            self._refresh_reduced_coocc()

        # Concatenate known queries and last results
        prediction = {
            td: [kw] for td, kw in self.known_queries.items() if td in trapdoor_list
        }
        prediction.update(dict(final_results))

        # Reset the known queries
        self.known_queries = old_known
        self._refresh_reduced_coocc()

        return prediction

    def base_accuracy(self, k=1, eval_dico=None):
        """Compute the base accuracy on the evaluation dictionary.
        No refinement. No clustering.

        Keyword Arguments:
            k {int} -- size of the prediction lists (default: {1})
            eval_dico {dict[trapdoor, keyword]} -- Evaluation dictionary (default: {None})

        Returns:
            Tuple[float, dict[trapdoor, List[keyword]]] -- a tuple containing the accuracy and the prediction dictionary
        """
        assert k > 0
        assert self.td_voc_info

        if eval_dico is None:
            logger.warning("Experimental setup is enabled.")
            # Trapdoor vocabulary is not a list of tokens but directly the list of keywords.
            local_eval_dico = {kw: kw for kw in self.td_voc_info.keys()}
        else:
            local_eval_dico = eval_dico
            # Keys: Trapdoors; Values: Keywords
        eval_trapdoors = list(local_eval_dico.keys())

        res_dict = self.tf_predict(trapdoor_list=eval_trapdoors, k=k)

        match_list = [
            eval_kw in res_dict[eval_td] for eval_td, eval_kw in local_eval_dico.items()
        ]
        recovery_rate = sum(match_list) / len(local_eval_dico)
        return recovery_rate, res_dict

    def queryvolution_accuracy(self, eval_dico=None, cluster_max_size: int = 10, ref_speed: int = 0):
        """Compute the QueRyvolution accuracy on the evaluation dictionary.

        Keyword Arguments:
            eval_dico {dict[trapdoor, keyword]} -- Evaluation dictionary (default: {None})

        Returns:
            Tuple[float, dict[trapdoor, List[keyword]]] -- a tuple containing the accuracy and the prediction dictionary
        """
        assert self.td_voc_info

        if eval_dico is None:
            logger.warning("Experimental setup is enabled.")
            # Trapdoor vocabulary is not a list of tokens but directly the list of keywords.
            local_eval_dico = {kw: kw for kw in self.td_voc_info.keys()}
        else:
            local_eval_dico = eval_dico
            # Keys: Trapdoors; Values: Keywords
        eval_trapdoors = list(local_eval_dico.keys())

        res_dict = self.tf_predict_with_refinement(
            trapdoor_list=eval_trapdoors,
            cluster_max_size=cluster_max_size,
            ref_speed=ref_speed
        )

        match_list = [
            eval_kw in res_dict[eval_td] for eval_td, eval_kw in local_eval_dico.items()
        ]
        recovery_rate = sum(match_list) / len(local_eval_dico)
        return recovery_rate, res_dict

    def queryvolution_accuracy_no_clustering(self, eval_dico=None, cluster_max_size: int = 1, ref_speed: int = 0):
        """
        Compute the QueRyvolution accuracy on the evaluation dictionary.

        Keyword Arguments:
            eval_dico {dict[trapdoor, keyword]} -- Evaluation dictionary (default: {None})

        Returns:
            Tuple[float, dict[trapdoor, List[keyword]]] -- a tuple containing the accuracy and the prediction dictionary
        """
        return self.queryvolution_accuracy(eval_dico=eval_dico, cluster_max_size=cluster_max_size, ref_speed=ref_speed)


def best_candidate_clustering_tf_optimized(
        sorted_indices: tf.Tensor,
        sorted_scores: tf.Tensor,
        cluster_max_size: int = 10,
        cluster_min_sensitivity: float = 0.0,
        include_score: bool = False,
        include_cluster_sep: bool = False,
):
    """From a tensor of scores, extracts the best-candidate cluster Smax using
    simple-linkage clustering.

    Arguments:
        sorted_indices {<tf.Tensor>} -- List of the keyword vocabulary indices that correspond to the scores
        sorted_scores {<tf.Tensor, dtype=DATA_TYPE>} -- Sorted tensor of the scores

    Keyword Arguments:
        cluster_max_size {int} -- maximum size of the prediction clusters (default: {10})
        cluster_min_sensitivity {float} -- minimum leap size. Otherwise returns the
                                            maximum-size cluster (default: {0.0})
        include_cluster_sep {bool} -- if True, returns also the cluster separation
                                    from the rest of the points (default: {False})
        include_score {bool} -- if True, returns also the score for each keyword
                                in the clusters (default: {False})

    Returns:
        Tuple[tf.Tensor,tf.Tensor,float] -- Tuple containing the cluster of keyword indices, keyword scores and the
                                            cluster separation if include_cluster_sep == True
    """
    # logger.debug(f"Sorted indices: {sorted_indices}, shape: {sorted_indices.shape[0]}")
    # logger.debug(f"Sorted  scores: {sorted_scores}, shape: {sorted_scores.shape[0]}")

    # Pick that 'cluster_max_size + 1' last items from tensors
    sorted_indices = tf.slice(
        sorted_indices,
        [sorted_indices.shape[0] - cluster_max_size - 1],
        [cluster_max_size + 1],
        name="AM_clustering_sorted_indices_slice"
    )
    sorted_scores = tf.slice(
        sorted_scores,
        [sorted_scores.shape[0] - cluster_max_size - 1],
        [cluster_max_size + 1],
        name="AM_clustering_sorted_scores_slice"
    )

    # Get the slices to calculate the difference
    sorted_scores_minus_first = tf.slice(
        sorted_scores,
        [1],
        [sorted_scores.shape[0] - 1],
        name="AM_clustering_sorted_scores_minus_first_slice"
    )
    sorted_scores_offset_1 = tf.slice(
        sorted_scores,
        [0],
        [sorted_scores.shape[0] - 1],
        name="AM_clustering_sorted_scores_offset_one_slice"
    )

    # Calculate index with maxed difference
    diff_list = tf.math.subtract(
        sorted_scores_minus_first,
        sorted_scores_offset_1,
        name="AM_clustering_subtract_minus_first_offset_one"
    )

    # Find index of max value
    maxed_ind = tf.math.argmax(diff_list, name="AM_clustering_argmax_diff_list")

    maximum_leap = diff_list[maxed_ind]     # Maxed difference
    ind_max_leap = maxed_ind + 1            # Index with maxed difference

    if maximum_leap > cluster_min_sensitivity:
        # Pick the last elements starting from 'ind_max_leap'
        best_candidates_indices = tf.slice(
            sorted_indices,
            [ind_max_leap],
            [sorted_indices.shape[0] - ind_max_leap],
            name="AM_clustering_maximum_leap_sorted_indices_slice"
        )
        if include_score:
            best_candidates_scores = tf.slice(
                sorted_scores,
                [ind_max_leap],
                [sorted_scores.shape[0] - ind_max_leap],
                name="AM_clustering_include_score_sorted_scores_slice"
            )
            best_candidates = best_candidates_indices, best_candidates_scores
        else:
            best_candidates = (best_candidates_indices,)
    else:
        # Pick the last 'cluster_max_size' items
        best_candidates_indices = tf.slice(
            sorted_indices,
            [sorted_indices.shape[0] - cluster_max_size],
            [cluster_max_size],
            name="AM_clustering_cluster_max_size_sorted_indices_slice"
        )
        if include_score:
            best_candidates_scores = tf.slice(
                sorted_scores,
                [sorted_indices.shape[0] - cluster_max_size],
                [cluster_max_size],
                name="AM_clustering_include_score_sorted_scores slice"
            )
            best_candidates = best_candidates_indices, best_candidates_scores
        else:
            best_candidates = (best_candidates_indices,)

    # logger.debug(f"Sorted scores: {sorted_scores}")
    # logger.debug(f"Maximum leap: {maximum_leap}; Cluster_min_sensitivity: {cluster_min_sensitivity}")
    if include_cluster_sep:
        best_candidates += (maximum_leap,)
        return best_candidates
    else:
        return best_candidates


def _sub_pred_tf_optimized_multi_core(
    _ind,
    td_voc_info,
    td_reduced_coocc,
    kw_reduced_coocc,
    td_list,
    k=0,
    cluster_max_size=0,
    cluster_min_sensitivity=0,
    include_score=False,
    include_cluster_sep=False,
    norm_ord='euclidean',
) -> list:
    """
    Helper function to calculate predicted score using multiple cores.

    Returns: List[Tuple[Trapdoor, PredictedKeyword(s), Optional(cluster_score)]]
    """
    prediction = []
    start_time = time.time()

    for trapdoor in tqdm.tqdm(
            iterable=td_list,
            total=len(td_list),
            desc=f"Prediction for one trapdoor ({_ind})"):
        predicted_result = _sub_pred_tf_optimized_one_trapdoor_single_core(
            _ind,
            td_voc_info=td_voc_info,
            td_reduced_coocc=td_reduced_coocc,
            kw_reduced_coocc=kw_reduced_coocc,
            trapdoor=trapdoor,
            k=k,
            cluster_max_size=cluster_max_size,
            cluster_min_sensitivity=cluster_min_sensitivity,
            include_score=include_score,
            include_cluster_sep=include_cluster_sep,
            norm_ord=norm_ord,
        )
        prediction.append(predicted_result)
    duration = time.time() - start_time
    print(f"Sub prediction {_ind}, took: {duration}")
    return prediction


def _sub_pred_tf_optimized_one_trapdoor_single_core(
    _ind,
    td_voc_info,
    td_reduced_coocc,
    kw_reduced_coocc,
    trapdoor,
    k=0,
    cluster_max_size=0,
    cluster_min_sensitivity=0,
    include_score=False,
    include_cluster_sep=False,
    norm_ord='euclidean',
) -> tuple:
    """
    Sub function to calculate the prediction(s) for one trapdoor.

    Returns: Tuple[Trapdoor, [Predictions], Optional(cluster_score)]
    """
    try:
        trapdoor_ind = td_voc_info[trapdoor]["vector_ind"]
    except KeyError:
        logger.debug(f"Unknown trapdoor: {trapdoor}")
        return ((trapdoor, [], 0) if include_cluster_sep else (trapdoor, []))

    trapdoor_vec = td_reduced_coocc[trapdoor_ind]

    run_times = {}

    # Tensor of all keyword vectors minus the trapdoor vector
    start_time = time.time()
    trapdoor_vector_differences = tf.subtract(kw_reduced_coocc, trapdoor_vec, name="AM_kw_reduced_coocc_-_trapdoor_vec")
    run_times['differences_time'] = time.time() - start_time

    # Distance between keyword vectors and trapdoor per vector
    start_time = time.time()
    td_kws_distances = tf.norm(trapdoor_vector_differences, ord=norm_ord, axis=1, name="AM_norm")
    run_times['kws_distances_time'] = time.time() - start_time

    # Calculate score with natural log
    start_time = time.time()
    td_kws_scores = -tf.math.log(td_kws_distances, name="AM_log(td_kws_distances)")
    run_times['kws_scores_time'] = time.time() - start_time

    # Tensorflow arg sort will give us the order of indices that satisfy an ascending order of scores
    start_time = time.time()
    scores_best_indices = tf.argsort(td_kws_scores, axis=-1, name="AM_argsort(td_kws_scores)")
    run_times['argsort_kws_time'] = time.time() - start_time

    start_time = time.time()
    td_kws_scores_sorted = tf.sort(td_kws_scores, axis=-1, name="AM_sort(td_kws_scores)")
    run_times['sort_kws_time'] = time.time() - start_time

    if cluster_max_size:  # Cluster mode
        # cluster = best_indices, scores_for_indices, maximum_leap
        start_time = time.time()
        cluster = best_candidate_clustering_tf_optimized(
            scores_best_indices,
            td_kws_scores_sorted,
            cluster_max_size=cluster_max_size,
            cluster_min_sensitivity=cluster_min_sensitivity,
            include_score=include_score,
            include_cluster_sep=include_cluster_sep,
        )
        run_times['clustering_time'] = time.time() - start_time

        # for name, elapsed_time in run_times.items():
        #     logger.debug(f"{name} took: {elapsed_time} s")

        # So appending: (trapdoor, best_indices, scores_for_indices, maximum_leap)
        #            or (trapdoor, best_indices, scores_for_indices)
        return (trapdoor, *cluster)
    else:  # k-highest-score mode
        best_candidates_indices = tf.slice(
            scores_best_indices,
            [scores_best_indices.shape[0] - k],
            [k],
            name="AM_k-score_slice_scores_indices"
        )
        best_candidates_scores = tf.slice(
            td_kws_scores_sorted,
            [td_kws_scores_sorted.shape[0] - k],
            [k],
            name="AM_k-score_slice_scores_kws"
        )

        if include_score:
            best_candidates = best_candidates_indices, best_candidates_scores
        else:
            best_candidates = (best_candidates_indices,)

        # So appending: (trapdoor, best_indices, scores_for_indices) or (trapdoor, best_indices)
        return (trapdoor, *best_candidates)
