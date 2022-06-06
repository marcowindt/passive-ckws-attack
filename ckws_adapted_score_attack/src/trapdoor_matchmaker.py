import gc
import math
import resource
import time

import colorlog
import psutil
import tensorflow as tf
import tqdm

from ckws_adapted_score_attack.src.config import DATA_TYPE, PREDICT_BATCHES, PROFILE_PREDICTION_K, \
    PROFILE_PREDICTION_REFINED, CLUSTERING_BATCHES, NUM_GPUS, PROFILE_STRATEGY, TRAPDOOR_BATCHES

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")

MAX_CPUS = 64


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE),
        tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE),
    ],
    autograph=True,
)
def test_sum_distance_euclidean_sub(
    _td_list_vector_indices: tf.Tensor,
    sub_kw_reduced_coocc: tf.Tensor,
    sub_td_reduced_coocc: tf.Tensor,
) -> tf.Tensor:
    # assert tf.shape(sub_kw_reduced_coocc)[1] == tf.shape(sub_td_reduced_coocc)[1], \
    #     "tf.shape(sub_kw_reduced_coocc)[1] != tf.shape(sub_kw_reduced_coocc)[1]"

    sum_td_kws_distances = tf.TensorArray(
        dtype=DATA_TYPE,
        size=tf.shape(_td_list_vector_indices)[0],
        dynamic_size=False,
        name="TM_init_tensorarray_sub_td_kws_distances"
    )

    idx = tf.constant(0)
    tf_while_cond = lambda i, x: tf.less(i, tf.shape(_td_list_vector_indices)[0])

    def tf_while_body(trapdoor_ind: tf.Tensor, sum_distances: tf.TensorArray):
        sub_trapdoor_vec = sub_td_reduced_coocc[_td_list_vector_indices[trapdoor_ind]]

        # Distance vector and sub kw coocc matrix
        sub_trapdoor_vector_differences = tf.subtract(
            sub_kw_reduced_coocc,
            sub_trapdoor_vec,
            name="TM_sub_kw_reduced_coocc_-_sub_trapdoor_vec"
        )

        # Square (as sub part for the euclidean distance, take the sqrt after all sub parts are done)
        squared_td_kws_distances = tf.square(
            sub_trapdoor_vector_differences,
            name="TM_square_sub_trapdoor_vector_differences"
        )

        # Sum (as sub part for the euclidean distance, sum along axis=1)
        sum_distances = sum_distances.write(
            trapdoor_ind,
            tf.reduce_sum(
                squared_td_kws_distances,
                axis=1,
                name="TM_reduce_sum_squared_td_kws_distances"
            ),
            name="TM_write_idx_sum_squares_in_TensorArray"
        )

        return tf.add(trapdoor_ind, 1, name="TM_add_idx_add_1_sub_td_list_loop"), sum_distances

    idx, result_sum_distances = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[idx, sum_td_kws_distances],
        parallel_iterations=24,
        name="TM_sum_distance_euclidean_sub_while_loop"
    )

    return result_sum_distances.stack(name="TM_stack_sum_td_kws_distances_tensorarray")


@tf.function(experimental_relax_shapes=True, experimental_follow_type_hints=True, autograph=True)
def predict_one_trapdoor_clustering(
        _ind: tf.Tensor,
        trapdoor_vec: tf.Tensor,
        kw_reduced_coocc: tf.Tensor,
        cluster_max_size: tf.Tensor,
        cluster_min_sensitivity: tf.Tensor,
        # include_score: tf.Tensor,
        # include_cluster_sep: tf.Tensor,
):
    """
    Sub function to calculate the prediction(s) for one trapdoor.

    Returns: Tuple[Trapdoor, [Predictions], Optional(cluster_score)]
    """
    # Tensor of all keyword vectors minus the trapdoor vector
    trapdoor_vector_differences = tf.subtract(kw_reduced_coocc, trapdoor_vec, name="TM_kw_reduced_coocc_-_trapdoor_vec")

    # Distance between keyword vectors and trapdoor per vector
    td_kws_distances = tf.norm(trapdoor_vector_differences, ord='euclidean', axis=1, name="TM_norm")

    # Calculate score with natural log
    td_kws_scores = -tf.math.log(td_kws_distances, name="TM_log_td_kws_distances")

    # Tensorflow arg sort will give us the order of indices that satisfy an ascending order of scores
    scores_best_indices = tf.argsort(td_kws_scores, axis=-1, name="TM_argsort_td_kws_scores")

    td_kws_scores_sorted = tf.sort(td_kws_scores, axis=-1, name="TM_sort_td_kws_scores")

    # cluster = best_indices, scores_for_indices, maximum_leap
    cluster = best_candidate_clustering(
        sorted_indices=scores_best_indices,
        sorted_scores=td_kws_scores_sorted,
        cluster_max_size=cluster_max_size,
        cluster_min_sensitivity=cluster_min_sensitivity,
        # include_score=include_score,
        # include_cluster_sep=include_cluster_sep,
    )

    # So appending: (trapdoor_ind, best_indices, scores_for_indices, maximum_leap)
    #            or (trapdoor_ind, best_indices, scores_for_indices)
    #            or (trapdoor_ind, best_indices, maximum_leap)
    #            or (trapdoor_ind, best_indices)
    return (_ind, *cluster)


@tf.function(experimental_relax_shapes=True, experimental_follow_type_hints=True, autograph=True)
def predict_one_trapdoor_k(
        _ind: tf.Tensor,
        trapdoor_vec: tf.Tensor,
        kw_reduced_coocc: tf.Tensor,
        k: tf.Tensor,
):
    """
    Sub function to calculate the prediction(s) for one trapdoor.

    Returns: Tuple[Trapdoor, [Predictions], Optional(cluster_score)]
    """
    # Tensor of all keyword vectors minus the trapdoor vector
    trapdoor_vector_differences = tf.subtract(kw_reduced_coocc, trapdoor_vec, name="TM_kw_reduced_coocc_-_trapdoor_vec")

    # Distance between keyword vectors and trapdoor per vector
    td_kws_distances = tf.norm(trapdoor_vector_differences, ord='euclidean', axis=1, name="TM_norm")

    # Calculate score with natural log
    td_kws_scores = -tf.math.log(td_kws_distances, name="TM_log_td_kws_distances")

    # Tensorflow arg sort will give us the order of indices that satisfy an ascending order of scores
    scores_best_indices = tf.argsort(td_kws_scores, axis=-1, name="TM_argsort_td_kws_scores")

    # td_kws_scores_sorted = tf.sort(td_kws_scores, axis=-1, name="TM_sort_td_kws_scores")

    best_candidates_indices = tf.slice(
        scores_best_indices,
        [tf.shape(scores_best_indices)[0] - k],
        [k],
        name="AM_k-score_slice_scores_indices"
    )

    # So appending: (trapdoor, best_indices, scores_for_indices) or (trapdoor, best_indices)
    return _ind, best_candidates_indices


@tf.function
def run_with_strategy(strategy, distributed_dataset, distributed_values, end_shape):
    def sum_distance_euclidean_sub(_td_list_vector_indices, ds) -> tf.Tensor:
        # assert tf.shape(sub_kw_reduced_coocc)[1] == tf.shape(sub_td_reduced_coocc)[1], \
        #     "tf.shape(sub_kw_reduced_coocc)[1] != tf.shape(sub_td_reduced_coocc)[1]"

        sub_kw_reduced_coocc, sub_td_reduced_coocc = ds

        sum_td_kws_distances = tf.TensorArray(
            dtype=DATA_TYPE,
            size=tf.shape(_td_list_vector_indices)[0],
            dynamic_size=False,
            name="TM_init_tensorarray_sub_td_kws_distances"
        )

        idx = tf.constant(0)
        tf_while_cond = lambda i, x: tf.less(i, tf.shape(_td_list_vector_indices)[0])

        def tf_while_body(trapdoor_ind: tf.Tensor, sum_distances: tf.TensorArray):
            sub_trapdoor_vec = sub_td_reduced_coocc[_td_list_vector_indices[trapdoor_ind]]

            # Distance vector and sub kw coocc matrix
            sub_trapdoor_vector_differences = tf.subtract(
                sub_kw_reduced_coocc,
                sub_trapdoor_vec,
                name="TM_sub_kw_reduced_coocc_-_sub_trapdoor_vec"
            )

            # Square (as sub part for the euclidean distance, take the sqrt after all sub parts are done)
            squared_td_kws_distances = tf.square(
                sub_trapdoor_vector_differences,
                name="TM_square_sub_trapdoor_vector_differences"
            )

            # Sum (as sub part for the euclidean distance, sum along axis=1)
            sum_distances = sum_distances.write(
                trapdoor_ind,
                tf.reduce_sum(
                    squared_td_kws_distances,
                    axis=1,
                    name="TM_reduce_sum_squared_td_kws_distances"
                ),
                name="TM_write_idx_sum_squares_in_TensorArray"
            )

            return tf.add(trapdoor_ind, 1, name="TM_add_idx_add_1_sub_td_list_loop"), sum_distances

        idx, result_sum_distances = tf.while_loop(
            cond=tf_while_cond,
            body=tf_while_body,
            loop_vars=[idx, sum_td_kws_distances],
            parallel_iterations=24,
            name="TM_sum_distance_euclidean_sub_while_loop"
        )

        return result_sum_distances.stack(name="TM_stack_sum_td_kws_distances_tensorarray")

    def run_prediction_step(inputs, ds_values):
        return strategy.run(
            fn=sum_distance_euclidean_sub,
            args=(inputs, ds_values)
        )

    t_arr_1 = tf.zeros(shape=(tf.cast(tf.math.ceil(end_shape[0] / 2), dtype=tf.int32), end_shape[1]), dtype=DATA_TYPE)
    t_arr_2 = tf.zeros(shape=(tf.cast(tf.math.floor(end_shape[0] / 2), dtype=tf.int32), end_shape[1]), dtype=DATA_TYPE)

    idx = tf.constant(0)
    for input_data in distributed_dataset:
        res = run_prediction_step(inputs=input_data, ds_values=distributed_values)

        sub_res = strategy.experimental_local_results(res)

        if idx == tf.constant(0):
            t_arr_1 = tf.add(t_arr_1, tf.concat(sub_res, axis=0))
        else:
            t_arr_2 = tf.add(t_arr_2, tf.concat(sub_res, axis=0))

        idx = tf.add(idx, 1)

    return tf.concat([t_arr_1, t_arr_2], axis=0)


def euclidean_distance_splitted_trapdoors(
    current_sub_kw_reduced_coocc: tf.Tensor,
    current_sub_td_reduced_coocc: tf.Tensor,
    td_list_vector_indices: tf.Tensor,
):
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    batch_size = (tf.math.ceil(tf.shape(td_list_vector_indices)[0] / 4)) * strategy.num_replicas_in_sync
    tds_dataset = tf.data.Dataset.from_tensor_slices(td_list_vector_indices).batch(tf.cast(batch_size, dtype=tf.int64))

    # Setting auto sharding OFF to prevent annoying warnings
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tds_dataset = tds_dataset.with_options(options)

    with strategy.scope():
        dist_dataset = strategy.experimental_distribute_dataset(tds_dataset)

        dist_values = (
            strategy.experimental_distribute_values_from_function(
                lambda _: tf.identity(current_sub_kw_reduced_coocc)
            ),
            strategy.experimental_distribute_values_from_function(
                lambda _: tf.identity(current_sub_td_reduced_coocc)
            ),
        )

    if PROFILE_STRATEGY:
        tf.profiler.experimental.start('profiler/log')

    result = run_with_strategy(
        strategy=strategy,
        distributed_dataset=dist_dataset,
        distributed_values=dist_values,
        end_shape=(tf.shape(td_list_vector_indices)[0], tf.shape(current_sub_kw_reduced_coocc)[0])
    )

    if PROFILE_STRATEGY:
        tf.profiler.experimental.stop()

    return result


def euclidean_distance_kw_coocc_trapdoors_distributed_strategy(
        td_list_vector_indices: tf.Tensor,
        kw_reduced_coocc: tf.Tensor,
        td_reduced_coocc: tf.Tensor,
        splits: tf.Tensor,
):
    with tf.device("/device:CPU:0"):
        result = tf.zeros(
            shape=(tf.shape(td_list_vector_indices)[0], tf.shape(kw_reduced_coocc)[0]),
            dtype=DATA_TYPE,
            name="TM_trapdoor_subs_td_kws_distances_zeros"
        )

    intermediate_results = []
    for idx in tf.range(PREDICT_BATCHES):
        with tf.device("/device:CPU:0"):
            common_split_size = splits[0]
            current_split_size = splits[idx]

            current_sub_kw_reduced_coocc = tf.slice(kw_reduced_coocc,
                                                    begin=[0, idx * common_split_size],
                                                    size=[tf.shape(kw_reduced_coocc)[0], current_split_size],
                                                    name="TM_kw_reduced_coocc_slice")

            current_sub_td_reduced_coocc = tf.slice(td_reduced_coocc,
                                                    begin=[0, idx * common_split_size],
                                                    size=[tf.shape(td_reduced_coocc)[0], current_split_size],
                                                    name="TM_td_reduced_coocc")

        sub_result = euclidean_distance_splitted_trapdoors(
            current_sub_kw_reduced_coocc=current_sub_kw_reduced_coocc,
            current_sub_td_reduced_coocc=current_sub_td_reduced_coocc,
            td_list_vector_indices=td_list_vector_indices,
        )

        intermediate_results.append(sub_result)

        with tf.device("/device:CPU:0"):
            result = tf.add(result, sub_result, name="TM_add_sub_result_to_result")
            del sub_result

    result = tf.add_n(intermediate_results)

    with tf.device("/device:CPU:0"):
        all_tds_distance_sums = result
        logger.debug(f"all_tds_distance_sums: {all_tds_distance_sums}")
        all_tds_distance_sqrt = tf.sqrt(all_tds_distance_sums, name="TM_sqrt_distance_scores")
        all_tds_distance_score = -tf.math.log(all_tds_distance_sqrt, name="TM_log_distance_scores")

    return all_tds_distance_score


def euclidean_distance_kw_coocc_trapdoors(
        td_list_vector_indices: tf.Tensor,
        kw_reduced_coocc: tf.Tensor,
        td_reduced_coocc: tf.Tensor,
        splits: tf.Tensor,
        tds_splits: tf.Tensor,
):
    with tf.device("/device:CPU:0"):
        result = tf.zeros(
            shape=(tf.shape(td_list_vector_indices)[0], tf.shape(kw_reduced_coocc)[0]),
            dtype=DATA_TYPE,
            name="TM_trapdoor_subs_td_kws_distances_zeros"
        )

    # intermediate_results = []
    for idx in tf.range(PREDICT_BATCHES):
        with tf.device("/device:CPU:0"):
            common_split_size = splits[0]
            current_split_size = splits[idx]

            current_sub_kw_reduced_coocc = tf.slice(kw_reduced_coocc,
                                                    begin=[0, idx * common_split_size],
                                                    size=[tf.shape(kw_reduced_coocc)[0], current_split_size],
                                                    name="TM_kw_reduced_coocc_slice")

            current_sub_td_reduced_coocc = tf.slice(td_reduced_coocc,
                                                    begin=[0, idx * common_split_size],
                                                    size=[tf.shape(td_reduced_coocc)[0], current_split_size],
                                                    name="TM_td_reduced_coocc")

        with tf.device("/device:GPU:0"):
            current_sub_kw_reduced_coocc = tf.identity(current_sub_kw_reduced_coocc)
            current_sub_td_reduced_coocc = tf.identity(current_sub_td_reduced_coocc)

        subs = []
        for i, tds_split in enumerate(tds_splits):
            with tf.device("/device:CPU:0"):
                current_tds_list = tf.slice(
                    td_list_vector_indices,
                    begin=[i * tds_splits[0]],
                    size=[tds_split],
                )

            with tf.device("/device:GPU:0"):
                current_tds_list = tf.identity(current_tds_list)

                sub_result = test_sum_distance_euclidean_sub(
                    _td_list_vector_indices=current_tds_list,
                    sub_kw_reduced_coocc=current_sub_kw_reduced_coocc,
                    sub_td_reduced_coocc=current_sub_td_reduced_coocc,
                )

                subs.append(sub_result)

            with tf.device("/device:CPU:0"):
                current_tds_list = tf.identity(current_tds_list)
                del current_tds_list

        with tf.device("/device:CPU:0"):
            current_sub_kw_reduced_coocc = tf.identity(current_sub_kw_reduced_coocc)
            current_sub_td_reduced_coocc = tf.identity(current_sub_td_reduced_coocc)

            del current_sub_kw_reduced_coocc
            del current_sub_td_reduced_coocc

            result = tf.add(result, tf.concat(subs, axis=0))

    with tf.device("/device:CPU:0"):
        all_tds_distance_sums = result

        # logger.debug(f"all_tds_distance_sums: {all_tds_distance_sums}")

        all_tds_distance_sqrt = tf.sqrt(all_tds_distance_sums, name="TM_sqrt_distance_scores")

        all_tds_distance_score = -tf.math.log(all_tds_distance_sqrt, name="TM_log_distance_scores")

    return all_tds_distance_score


def pred_sub_td_list_using_clustering(
        sub_tds_distance_scores: tf.Tensor,
        sub_td_list: list,
        cluster_max_size: int = 0,
        cluster_min_sensitivity: float = 0.0,
        include_score: bool = True,
        include_cluster_sep: bool = True,
) -> list:
    arg_sorted_tds_scores = tf.argsort(sub_tds_distance_scores, axis=-1, name="TM_argsort_sub_tds_distance_scores")
    sorted_tds_scores = tf.gather(
        sub_tds_distance_scores, arg_sorted_tds_scores, axis=1, batch_dims=1, name="TM_gather_sub_tds_distance_scores"
    )

    sub_predictions = []
    for i, trapdoor in enumerate(sub_td_list):
        cluster = best_candidate_clustering(
            sorted_indices=arg_sorted_tds_scores[i],
            sorted_scores=sorted_tds_scores[i],
            cluster_max_size=cluster_max_size,
            cluster_min_sensitivity=cluster_min_sensitivity,
            include_score=include_score,
            include_cluster_sep=include_cluster_sep,
        )

        sub_predictions.append((trapdoor.numpy().decode('utf-8'), *cluster))

    return sub_predictions


def pred_sub_td_list_using_k(
        sub_tds_distance_scores: tf.Tensor,
        sub_td_list: list,
        k: int = 0,
        include_score: bool = False,
):
    arg_sorted_tds_scores = tf.argsort(sub_tds_distance_scores, axis=-1)
    sorted_tds_scores = tf.gather(sub_tds_distance_scores, arg_sorted_tds_scores, axis=1, batch_dims=1)

    sub_predictions = []
    for i, trapdoor in enumerate(sub_td_list):
        best_candidates_indices = tf.slice(arg_sorted_tds_scores, [arg_sorted_tds_scores.shape[0] - k], [k])
        best_candidates_scores = tf.slice(sorted_tds_scores, [sorted_tds_scores.shape[0] - k], [k])

        if include_score:
            best_candidates = best_candidates_indices, best_candidates_scores
        else:
            best_candidates = (best_candidates_indices,)

        # So appending: (trapdoor, best_indices, scores_for_indices) or (trapdoor, best_indices)
        sub_predictions.append((trapdoor, *best_candidates))


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=DATA_TYPE),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=DATA_TYPE),
    ],
    autograph=True
)
def best_candidate_clustering(
        sorted_indices: tf.Tensor,
        sorted_scores: tf.Tensor,
        cluster_max_size: tf.Tensor,
        cluster_min_sensitivity: tf.Tensor,
        # include_score: tf.Tensor, FALSE
        # include_cluster_sep: tf.Tensor, TRUE
):
    """From a tensor of scores, extracts the best-candidate cluster Smax using
    simple-linkage clustering.

    IMPORTANT: This version of clustering only includes cluster sep (thus NOT include score)

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
        [tf.shape(sorted_indices)[0] - cluster_max_size - tf.constant(1)],
        [cluster_max_size + tf.constant(1)],
        name="TM_clustering_sorted_indices_slice"
    )
    sorted_scores = tf.slice(
        sorted_scores,
        [tf.shape(sorted_scores)[0] - cluster_max_size - tf.constant(1)],
        [cluster_max_size + tf.constant(1)],
        name="TM_clustering_sorted_scores_slice"
    )

    # Get the slices to calculate the difference
    sorted_scores_minus_first = tf.slice(
        sorted_scores,
        [tf.constant(1)],
        [tf.shape(sorted_scores)[0] - tf.constant(1)],
        name="TM_clustering_sorted_scores_minus_first_slice"
    )
    sorted_scores_offset_1 = tf.slice(
        sorted_scores,
        [tf.constant(0)],
        [tf.shape(sorted_scores)[0] - tf.constant(1)],
        name="TM_clustering_sorted_scores_offset_1_slice"
    )

    # Calculate index with maxed difference
    diff_list = tf.math.subtract(
        sorted_scores_minus_first,
        sorted_scores_offset_1,
        name="TM_clustering_subtract_minus_first_offset_1"
    )

    # Find index of max value
    maxed_ind = tf.math.argmax(diff_list, output_type=tf.int32, name="TM_clustering_argmax_diff_list")

    maximum_leap = diff_list[maxed_ind]  # Maxed difference
    ind_max_leap = maxed_ind + tf.constant(1)  # Index with maxed difference

    if tf.greater(maximum_leap, cluster_min_sensitivity):
        # Pick the last elements starting from 'ind_max_leap'
        best_candidates_indices = tf.slice(
            sorted_indices,
            [ind_max_leap],
            [tf.shape(sorted_indices)[0] - ind_max_leap],
            name="TM_clustering_maximum_leap_sorted_indices_slice"
        )

        return best_candidates_indices, maximum_leap
    else:
        # Pick the last 'cluster_max_size' items
        best_candidates_indices = tf.slice(
            sorted_indices,
            [tf.shape(sorted_indices)[0] - cluster_max_size],
            [cluster_max_size],
            name="TM_clustering_cluster_max_size_sorted_indices_slice"
        )

        return best_candidates_indices, maximum_leap


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=DATA_TYPE),
    ],
    autograph=True)
def pred_td_list_using_clustering(
        tds_distance_scores: tf.Tensor,
        nb_trapdoors: tf.Tensor,
        cluster_max_size: tf.Tensor,
        cluster_min_sensitivity: tf.Tensor,
        # include_score: tf.Tensor,
        # include_cluster_sep: tf.Tensor,
):
    arg_sorted_tds_scores = tf.argsort(
        tds_distance_scores,
        axis=-1,
        name="TM_argsort_tds_distance_scores"
    )
    sorted_tds_scores = tf.gather(
        tds_distance_scores,
        arg_sorted_tds_scores,
        axis=1,
        batch_dims=1,
        name="TM_gather_sub_tds_distance_scores"
    )

    init_td_arr_indices = tf.TensorArray(dtype=tf.int32, size=nb_trapdoors, dynamic_size=False)
    init_td_arr_leaps = tf.TensorArray(dtype=DATA_TYPE, size=nb_trapdoors, dynamic_size=False)

    idx = tf.constant(0)
    tf_while_cond = lambda i, x, y: tf.less(i, nb_trapdoors)

    def tf_while_body(i: tf.Tensor, td_arr_indices: tf.TensorArray, td_arr_leaps: tf.TensorArray):
        best_candidates_indices, maximum_leap = best_candidate_clustering(
            sorted_indices=arg_sorted_tds_scores[i],
            sorted_scores=sorted_tds_scores[i],
            cluster_max_size=cluster_max_size,
            cluster_min_sensitivity=cluster_min_sensitivity,
            # include_score=include_score, FALSE
            # include_cluster_sep=include_cluster_sep, TRUE
        )

        return tf.add(i, 1), td_arr_indices.write(i, best_candidates_indices), td_arr_leaps.write(i, maximum_leap)

    result_i, result_td_arr_indices, result_td_arr_leaps = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[idx, init_td_arr_indices, init_td_arr_leaps],
        # shape_invariants=[idx.get_shape(), ]
        parallel_iterations=MAX_CPUS,
        name="TM_sum_distance_euclidean_sub_while_loop"
    )

    result_td_arr_indices = result_td_arr_indices.stack(name="TM_stack_td_arr_indices")
    result_td_arr_leaps = result_td_arr_leaps.stack(name="TM_stack_td_arr_leaps")

    return result_td_arr_indices, result_td_arr_leaps


@tf.function(experimental_relax_shapes=True, experimental_follow_type_hints=True, autograph=True)
def predict_trapdoors_loop(
        td_list_vector_indices: tf.Tensor,
        td_reduced_coocc: tf.Tensor,
        kw_reduced_coocc: tf.Tensor,
        cluster_max_size: tf.Tensor,
        cluster_min_sensitivity: tf.Tensor,
):
    i = tf.constant(0, name="TM_predict_initialize_idx_0_constant")
    tf_while_cond = lambda i, x, y: tf.less(i, tf.shape(td_list_vector_indices)[0], name="TM_while_cond_less_len_td_idx")

    init_td_arr_indices = tf.TensorArray(dtype=tf.int32, size=tf.shape(td_list_vector_indices)[0], dynamic_size=False)
    init_td_arr_leaps = tf.TensorArray(dtype=DATA_TYPE, size=tf.shape(td_list_vector_indices)[0], dynamic_size=False)

    def tf_while_body(td_idx: tf.Tensor, td_arr_indices: tf.TensorArray, td_arr_leaps: tf.TensorArray):
        td_ind, best_candidates_indices, maximum_leap = predict_one_trapdoor_clustering(
            _ind=td_list_vector_indices[td_idx],
            trapdoor_vec=td_reduced_coocc[td_list_vector_indices[td_idx]],
            kw_reduced_coocc=kw_reduced_coocc,
            cluster_max_size=cluster_max_size,
            cluster_min_sensitivity=cluster_min_sensitivity,
            # include_score=tf.constant(False),  # include_score,
            # include_cluster_sep=tf.constant(True),
        )

        add_one = tf.add(td_idx, 1, name="TM_while_loop_sum_distance_idx_add_1")

        return add_one, td_arr_indices.write(td_idx, best_candidates_indices), td_arr_leaps.write(td_idx, maximum_leap)

    idx, result_td_arr_indices, result_td_arr_leaps = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[i, init_td_arr_indices, init_td_arr_leaps],
        # shape_invariants=[i.get_shape()],
        parallel_iterations=MAX_CPUS,
        maximum_iterations=tf.shape(td_list_vector_indices)[0],
        swap_memory=False,
        name="TM_sum_distance_euclidean_sub_while_loop"
    )

    result_td_arr_indices = result_td_arr_indices.stack(name="TM_stack_td_arr_indices")
    # result_td_arr_scores = result_td_arr_scores.stack(name="TM_stack_td_arr_scores")
    result_td_arr_leaps = result_td_arr_leaps.stack(name="TM_stack_td_arr_leaps")

    return result_td_arr_indices, result_td_arr_leaps


def predict_trapdoors(
        td_voc_info,
        td_list: list,
        kw_reduced_coocc: tf.Tensor,
        td_reduced_coocc: tf.Tensor,
        cluster_max_size: int = 0,
        cluster_min_sensitivity: float = 0.0,
        include_score: bool = False,
        include_cluster_sep: bool = True,
) -> list:
    auto_garbage_collect()
    memory_usage()

    _td_list_vector_indices = []
    for trapdoor in td_list:
        try:
            _td_list_vector_indices.append(td_voc_info[trapdoor]["vector_ind"])
        except KeyError:
            # logger.debug(f"Unknown trapdoor: {trapdoor}")
            continue

    tf.profiler.experimental.start('profiler/log')
    td_list_vector_indices = tf.convert_to_tensor(
        _td_list_vector_indices,
        dtype=tf.int32,
        name="TM_convert_to_tensor_td_list_indices"
    )

    with tf.device("/device:GPU:0"):
        kw_reduced_coocc = tf.identity(kw_reduced_coocc)
        td_reduced_coocc = tf.identity(td_reduced_coocc)

        td_list_vector_indices = tf.identity(td_list_vector_indices)

        result_td_arr_indices, result_td_arr_leaps = predict_trapdoors_loop(
            td_list_vector_indices=td_list_vector_indices,
            kw_reduced_coocc=kw_reduced_coocc,
            td_reduced_coocc=td_reduced_coocc,
            cluster_max_size=tf.constant(cluster_max_size, dtype=tf.int32),
            cluster_min_sensitivity=tf.constant(cluster_min_sensitivity, dtype=DATA_TYPE),
        )

    with tf.device("/device:CPU:0"):
        td_list_vector_indices = tf.identity(td_list_vector_indices)

        kw_reduced_coocc = tf.identity(kw_reduced_coocc)
        td_reduced_coocc = tf.identity(td_reduced_coocc)

        result_td_arr_indices = tf.identity(result_td_arr_indices)
        result_td_arr_leaps = tf.identity(result_td_arr_leaps)

        del td_list_vector_indices

        del kw_reduced_coocc
        del td_reduced_coocc

    tf.keras.backend.clear_session()
    gc.collect()

    tf.profiler.experimental.stop()

    prediction = []
    for i, trapdoor in enumerate(td_list):
        prediction.append((trapdoor, result_td_arr_indices[i], result_td_arr_leaps[i]))

    memory_usage()

    return prediction


def predict_trapdoors_k(
        td_voc_info,
        td_list: list,
        kw_reduced_coocc: tf.Tensor,
        td_reduced_coocc: tf.Tensor,
        k: int = 0,
) -> list:
    auto_garbage_collect()
    memory_usage()

    _td_list_vector_indices = []
    for trapdoor in td_list:
        try:
            _td_list_vector_indices.append(td_voc_info[trapdoor]["vector_ind"])
        except KeyError:
            # logger.debug(f"Unknown trapdoor: {trapdoor}")
            continue
    td_list_vector_indices = tf.convert_to_tensor(
        _td_list_vector_indices,
        dtype=tf.int32,
        name="TM_convert_to_tensor_td_list_indices"
    )

    if PROFILE_PREDICTION_K:
        tf.profiler.experimental.start('profiler/log')

    with tf.device("/device:GPU:0"):
        kw_reduced_coocc = tf.identity(kw_reduced_coocc)
        td_reduced_coocc = tf.identity(td_reduced_coocc)

        td_list_vector_indices = tf.identity(td_list_vector_indices)

        result_td_arr_indices = predict_trapdoors_loop_k(
            td_list_vector_indices=td_list_vector_indices,
            kw_reduced_coocc=kw_reduced_coocc,
            td_reduced_coocc=td_reduced_coocc,
            k=tf.constant(k),
        )

    with tf.device("/device:CPU:0"):
        kw_reduced_coocc = tf.identity(kw_reduced_coocc)
        td_reduced_coocc = tf.identity(td_reduced_coocc)

        td_list_vector_indices = tf.identity(td_list_vector_indices)

        result_td_arr_indices = tf.identity(result_td_arr_indices)

        del kw_reduced_coocc
        del td_reduced_coocc

        del td_list_vector_indices

    if PROFILE_PREDICTION_K:
        tf.profiler.experimental.stop()

    prediction = []
    for i, trapdoor in enumerate(td_list):
        prediction.append((trapdoor, result_td_arr_indices[i]))

    memory_usage()

    return prediction


@tf.function(experimental_relax_shapes=True, experimental_follow_type_hints=True, autograph=True)
def predict_trapdoors_loop_k(
        td_list_vector_indices: tf.Tensor,
        td_reduced_coocc: tf.Tensor,
        kw_reduced_coocc: tf.Tensor,
        k: tf.Tensor,
):
    i = tf.constant(0, name="TM_predict_initialize_idx_0_constant")
    tf_while_cond = lambda i, x: tf.less(i, tf.shape(td_list_vector_indices)[0], name="TM_while_cond_less_len_td_idx")

    init_td_arr_indices = tf.TensorArray(dtype=tf.int32, size=tf.shape(td_list_vector_indices)[0], dynamic_size=False)

    def tf_while_body(td_idx: tf.Tensor, td_arr_indices: tf.TensorArray):
        td_ind, best_candidates_indices = predict_one_trapdoor_k(
            _ind=td_list_vector_indices[td_idx],
            trapdoor_vec=td_reduced_coocc[td_list_vector_indices[td_idx]],
            kw_reduced_coocc=kw_reduced_coocc,
            k=k
        )

        add_one = tf.add(td_idx, 1, name="TM_while_loop_sum_distance_idx_add_1")

        return add_one, td_arr_indices.write(td_idx, best_candidates_indices)

    idx, result_td_arr_indices = tf.while_loop(
        cond=tf_while_cond,
        body=tf_while_body,
        loop_vars=[i, init_td_arr_indices],
        # shape_invariants=[i.get_shape()],
        parallel_iterations=MAX_CPUS,
        name="TM_sum_distance_euclidean_sub_while_loop"
    )

    result_td_arr_indices = result_td_arr_indices.stack(name="TM_stack_td_arr_indices")
    # result_td_arr_scores = result_td_arr_scores.stack(name="TM_stack_td_arr_scores")
    # result_td_arr_leaps = result_td_arr_leaps.stack(name="TM_stack_td_arr_leaps")

    return result_td_arr_indices


def pred_td_list(
        td_voc_info,
        td_list: list,
        kw_reduced_coocc: tf.Tensor,
        td_reduced_coocc: tf.Tensor,
        cluster_max_size: int = 0,
        cluster_min_sensitivity: float = 0.0,
        include_score: bool = False,
        include_cluster_sep: bool = False,
):
    auto_garbage_collect()

    td_list_vector_indices = []
    for trapdoor in td_list:
        try:
            td_list_vector_indices.append(td_voc_info[trapdoor]["vector_ind"])
        except KeyError:
            # logger.debug(f"Unknown trapdoor: {trapdoor}")
            continue
    td_list_vector_indices = tf.convert_to_tensor(td_list_vector_indices, name="TM_convert_to_tensor_td_list_indices")

    batch_splits = tf.constant(
        _generate_splits_by_number_of_batches(axis_length=kw_reduced_coocc.shape[1], batches=PREDICT_BATCHES),
        name="TM_batch_size_coocc",
    )

    logger.debug(f"batch_splits: {batch_splits}")

    _batch_tds_splits = _generate_splits_by_number_of_batches(
        tf.shape(td_list_vector_indices)[0].numpy(), batches=TRAPDOOR_BATCHES)

    logger.debug(f"_batch_tds_splits: {_batch_tds_splits}")

    batch_tds_splits = tf.constant(
        _batch_tds_splits,
        name="TM_convert_to_tensor_tds_splits",
        dtype=tf.int32,
    )

    logger.debug(f"batch_tds_splits: {batch_tds_splits}")

    logger.debug(f"Start euclidean distance trapdoors procedure")

    if PROFILE_PREDICTION_REFINED:
        tf.profiler.experimental.start('profiler/log')

    start_time = time.time()

    tf.debugging.set_log_device_placement(True)
    all_tds_distance_score = euclidean_distance_kw_coocc_trapdoors(
        td_list_vector_indices=tf.constant(td_list_vector_indices),
        kw_reduced_coocc=kw_reduced_coocc,
        td_reduced_coocc=td_reduced_coocc,
        # Split tensors in number of batches (i.e. split along columns)
        splits=batch_splits,
        tds_splits=batch_tds_splits,
    )
    tf.debugging.set_log_device_placement(False)

    all_tds_distance_score_time = time.time()
    logger.debug(f"All tds #{len(td_list)} distance score took: {all_tds_distance_score_time - start_time} s")
    memory_usage()

    auto_garbage_collect()
    tf.keras.backend.clear_session()

    if PROFILE_PREDICTION_REFINED:
        tf.profiler.experimental.stop()

    logger.debug(f"all_tds_distance_score.shape: {tf.shape(all_tds_distance_score)}")

    if PROFILE_PREDICTION_REFINED:
        tf.profiler.experimental.start('profiler/log')

    with tf.device("/device:CPU:0"):
        tds_splits = _generate_splits_by_number_of_batches(
            tf.shape(td_list_vector_indices)[0],
            batches=CLUSTERING_BATCHES
        )

        subs_indices = []
        subs_leaps = []

        for i, tds_split in tqdm.tqdm(
            iterable=enumerate(tds_splits),
            desc="batch predictions",
            total=len(tds_splits),
        ):
            sub_all_tds_distance_score = tf.slice(
                all_tds_distance_score,
                begin=[i * tds_splits[0], 0],
                size=[tds_split, tf.shape(all_tds_distance_score)[1]],
                name="TM_slice_all_tds_distance_score",
            )

            with tf.device(f"/device:GPU:{i % NUM_GPUS}"):
                sub_result_td_arr_indices, sub_result_td_arr_leaps = pred_td_list_using_clustering(
                    tds_distance_scores=tf.constant(sub_all_tds_distance_score),
                    nb_trapdoors=tf.constant(tds_split),
                    cluster_max_size=tf.constant(cluster_max_size, dtype=tf.int32),
                    cluster_min_sensitivity=tf.constant(cluster_min_sensitivity, dtype=DATA_TYPE),
                    # include_score=tf.constant(include_score), FALSE
                    # include_cluster_sep=tf.constant(include_cluster_sep), TRUE
                )

            with tf.device("/device:CPU:0"):
                sub_result_td_arr_indices = tf.identity(sub_result_td_arr_indices)
                sub_result_td_arr_leaps = tf.identity(sub_result_td_arr_leaps)

                subs_indices.append(sub_result_td_arr_indices)
                subs_leaps.append(sub_result_td_arr_leaps)

    with tf.device("/device:CPU:0"):
        result_td_arr_indices = tf.concat(subs_indices, axis=0, name="TM_concat_subs_indices")
        result_td_arr_leaps = tf.concat(subs_leaps, axis=0, name="TM_concat_subs_leaps")

    prediction = []
    for i, trapdoor in enumerate(td_list):
        prediction.append((trapdoor, result_td_arr_indices[i], result_td_arr_leaps[i]))

    after_all_prediction_time = time.time()
    logger.debug(f"All tds #{len(td_list)} predict took: {after_all_prediction_time - all_tds_distance_score_time} s")
    memory_usage()

    if PROFILE_PREDICTION_REFINED:
        tf.profiler.experimental.stop()

    return prediction


def _generate_splits_by_number_of_cpu(axis_length: int):
    return _generate_splits_by_number_of_batches(axis_length=axis_length, batches=MAX_CPUS)


def _generate_splits_by_number_of_batches(axis_length: int, batches: int):
    split_size = int(math.ceil(axis_length / batches))
    too_much = batches * split_size - axis_length
    splits = [split_size for _ in range(batches)]

    while too_much / split_size > 1:
        too_much -= split_size
        splits.pop()

    splits[-1] -= too_much

    if splits[-1] == 0:
        splits.pop()

    return splits


def auto_garbage_collect(pct=40.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return


def memory_usage():
    logger.debug(f"Memory usage: {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}")
