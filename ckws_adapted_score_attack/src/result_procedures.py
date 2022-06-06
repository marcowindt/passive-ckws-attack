"""Functions used to produce the results presented in the paper.
"""
import csv
import gc
import hashlib
import multiprocessing
import random

import colorlog
import numpy as np
from scipy.special import comb
import tensorflow as tf
import time

from ckws_adapted_score_attack.src.conjunctive_extraction import ConjunctiveExtractor, generate_trapdoors, generate_known_queries
from ckws_adapted_score_attack.src.conjunctive_matchmaker import ConjunctiveScoreAttack
from ckws_adapted_score_attack.src.conjunctive_query_generator import ConjunctiveQueryResultExtractor
from ckws_adapted_score_attack.src.common import KeywordExtractor, conjunctive_keyword_combinations_indices
from ckws_adapted_score_attack.src.common import generate_known_queries as old_generate_known_queries
from ckws_adapted_score_attack.src.email_extraction import (
    split_df,
    extract_sent_mail_contents,
    extract_apache_ml,
    load_enron, load_preprocessed_dataset,
    load_preprocessed_enron, load_preprocessed_enron_float32,
)
from ckws_adapted_score_attack.src.query_generator import (
    QueryResultExtractor
)
from ckws_adapted_score_attack.src.matchmaker import ScoreAttack
from ckws_adapted_score_attack.src.trapdoor_matchmaker import memory_usage


MAX_CPUS = 64    # multiprocessing.cpu_count() if multiprocessing.cpu_count() <= 128 else 128

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")
NB_REP = 50
kw_conjunction_size = 2
start_time = time.time()


def understand_variance(result_file=f"{kw_conjunction_size}-kws_variance_understanding-{start_time}.csv"):
    with tf.device("/device:CPU:0"):
        similar_voc_size = 46
        server_voc_size = 46
        queryset_size = 150
        nb_known_queries = 5

        logger.debug(f"Server vocabulary size (artificial_kw_size: {kw_conjunction_size}): {server_voc_size}")
        logger.debug(f"Similar vocabulary size (artificial_kw_size: {kw_conjunction_size}): {similar_voc_size}")

        documents = extract_sent_mail_contents()

        with open(result_file, "w", newline="") as csv_file:
            fieldnames = [
                "Setup",
                "Nb similar docs",
                "Nb server docs",
                "Similar voc size",
                "Server voc size",
                "Nb queries seen",
                "Nb queries known",
                "QueRyvolution Acc",
            ]
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()
            for i in range(NB_REP):
                logger.info(f"Experiment {i+1} out of {NB_REP}")

                # Split similar and real dataset
                similar_docs, stored_docs = split_df(d_frame=documents, frac=0.4)

                logger.info("Extracting keywords from similar documents.")
                similar_extractor = KeywordExtractor(
                    similar_docs,
                    similar_voc_size,
                    min_freq=1,
                    with_kw_conjunction_size=kw_conjunction_size
                )

                logger.info("Extracting keywords from stored documents.")
                real_extractor = QueryResultExtractor(
                    stored_docs,
                    server_voc_size,
                    min_freq=1,
                    with_artificial_kw_size=kw_conjunction_size
                )

                logger.info(f"Generating {queryset_size} queries from stored documents")
                query_array, query_voc_plain = real_extractor.get_fake_queries(queryset_size)

                # Delete real extractor for memory reasons
                del real_extractor

                logger.debug(
                    f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
                )

                known_queries = old_generate_known_queries(  # Extracted with uniform law
                    similar_wordlist=similar_extractor.get_sorted_voc().numpy(),
                    stored_wordlist=query_voc_plain,
                    nb_queries=nb_known_queries,
                )

                logger.debug("Hashing the keywords of the stored documents (transforming them into trapdoor tokens)")

                # Trapdoor token == convey no information about the corresponding keyword
                temp_voc = []
                temp_known = {}
                eval_dict = {}  # Keys: Trapdoor tokens; Values: Keywords
                for keyword in query_voc_plain:
                    # We replace each keyword of the trapdoor dictionary by its hash
                    # So the matchmaker truly ignores the keywords behind the trapdoors.
                    fake_trapdoor = hashlib.sha1("".join(keyword).encode("utf-8")).hexdigest()
                    temp_voc.append(fake_trapdoor)
                    if known_queries.get(keyword):
                        temp_known[fake_trapdoor] = keyword
                    eval_dict[fake_trapdoor] = keyword
                query_voc = temp_voc
                known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

                matchmaker = ScoreAttack(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=query_voc,
                    known_queries=known_queries,
                )

                # Difference of known queries and fake queries are the trapdoor to recover
                td_list = list(set(eval_dict.keys()).difference(matchmaker.known_queries.keys()))

                refinement_results = matchmaker.predict_with_refinement(td_list, cluster_max_size=10, ref_speed=10)
                refinement_accuracy = np.mean(
                    [eval_dict[td] in candidates for td, candidates in refinement_results.items()]
                )

                writer.writerow(
                    {
                        "Setup": "Normal",
                        "Nb similar docs": similar_docs.shape[0],
                        "Nb server docs": stored_docs.shape[0],
                        "Similar voc size": similar_voc_size,
                        "Server voc size": server_voc_size,
                        "Nb queries seen": queryset_size,
                        "Nb queries known": nb_known_queries,
                        "QueRyvolution Acc": refinement_accuracy,
                    }
                )

                logger.debug(
                    f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
                )

                stored_wordlist = query_voc_plain[: len(query_voc_plain) // 4]
                known_queries = generate_known_queries(  # Extracted with uniform law
                    similar_wordlist=similar_extractor.get_sorted_voc(),
                    stored_wordlist=stored_wordlist,
                    nb_queries=nb_known_queries,
                )

                logger.debug("Hashing the keywords of the stored documents (transforming them into trapdoor tokens)")

                # Trapdoor token == convey no information about the corresponding keyword
                temp_voc = []
                temp_known = {}
                eval_dict = {}  # Keys: Trapdoor tokens; Values: Keywords
                for keyword in query_voc_plain:
                    # We replace each keyword of the trapdoor dictionary by its hash
                    # So the matchmaker truly ignores the keywords behind the trapdoors.
                    fake_trapdoor = hashlib.sha1("".join(keyword).encode("utf-8")).hexdigest()
                    temp_voc.append(fake_trapdoor)
                    if known_queries.get(keyword):
                        temp_known[fake_trapdoor] = keyword
                    eval_dict[fake_trapdoor] = keyword
                query_voc = temp_voc
                known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

                matchmaker = ScoreAttack(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=query_voc,
                    known_queries=known_queries,
                )

                # Difference of known queries and fake queries are the trapdoors to recover
                td_list = list(set(eval_dict.keys()).difference(matchmaker.known_queries.keys()))

                refinement_results = matchmaker.predict_with_refinement(td_list, cluster_max_size=10, ref_speed=10)
                refinement_accuracy = np.mean(
                    [eval_dict[td] in candidates for td, candidates in refinement_results.items()]
                )

                writer.writerow(
                    {
                        "Setup": "Top 25%",
                        "Nb similar docs": similar_docs.shape[0],
                        "Nb server docs": stored_docs.shape[0],
                        "Similar voc size": similar_voc_size,
                        "Server voc size": server_voc_size,
                        "Nb queries seen": queryset_size,
                        "Nb queries known": nb_known_queries,
                        "QueRyvolution Acc": refinement_accuracy,
                    }
                )

                # Flush, such that intermediate results don't get lost if something bad happens
                csv_file.flush()


def cluster_size_statistics(result_file=f"{kw_conjunction_size}-kws_cluster_size-{start_time}.csv"):
    with tf.device("/device:CPU:0"):
        # Params
        similar_voc_size = 20
        server_voc_size = 20
        queryset_size = 150
        nb_known_queries = 15
        max_cluster_sizes = [1, 5, 10, 20, 30, 50]

        cluster_results = {
            max_size: {"accuracies": [], "cluster_sizes": []}
            for max_size in max_cluster_sizes
        }

        document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ = load_preprocessed_dataset(prefix='')

        # Split to N sparse tensors: one sparse tensor per email
        emails = tf.sparse.split(
            sp_input=document_keyword_occurrence,
            num_split=document_keyword_occurrence.dense_shape[0],
            axis=0,
        )

        logger.debug(f"Number of emails: {len(emails)}")

        keyword_combinations = None

        for i in range(NB_REP):
            logger.info(f"Experiment {i+1} out of {NB_REP}")

            # Split similar and real dataset
            email_ids = list(range(document_keyword_occurrence.dense_shape[0]))
            random.shuffle(email_ids)

            similar_doc_ids, stored_doc_ids = email_ids[:int(len(emails) * 0.4)], email_ids[int(len(emails) * 0.4):]
            logger.debug(f"Generated random email ids")

            similar_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in similar_doc_ids], axis=0)
            logger.debug(f"Similar docs shape: {similar_docs.dense_shape}")

            stored_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in stored_doc_ids], axis=0)
            logger.debug(f"Stored docs shape: {stored_docs.dense_shape}")

            # Extract keywords from similar dataset
            similar_extractor = ConjunctiveExtractor(
                occurrence_array=similar_docs,
                keyword_voc=sorted_keyword_voc,
                keyword_occ=sorted_keyword_occ,
                voc_size=similar_voc_size,
                kw_conjunction_size=kw_conjunction_size,
                min_freq=1,
                precalculated_artificial_keyword_combinations_indices=keyword_combinations,
                multi_core=False,
            )

            # Extract keywords from real dataset
            real_extractor = ConjunctiveQueryResultExtractor(
                stored_docs,
                sorted_keyword_voc,
                sorted_keyword_occ,
                server_voc_size,
                kw_conjunction_size,
                1,
                keyword_combinations,
                False,
            )

            logger.info(f"Generating {queryset_size} queries from stored documents")
            query_array, query_voc_plain = real_extractor.get_fake_queries(queryset_size)

            del real_extractor
            logger.debug(f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)")

            known_queries = old_generate_known_queries(  # Extracted with uniform law
                similar_wordlist=similar_extractor.get_sorted_voc().numpy(),
                stored_wordlist=query_voc_plain,
                nb_queries=nb_known_queries,
            )

            logger.debug("Hashing the keywords of the stored documents (transforming them into trapdoor tokens)")

            # Trapdoor token == convey no information about the corresponding keyword
            temp_voc = []
            temp_known = {}
            eval_dict = {}  # Keys: Trapdoor tokens; Values: Keywords

            for keyword in query_voc_plain:
                # We replace each keyword of the trapdoor dictionary by its hash
                # So the matchmaker truly ignores the keywords behind the trapdoors.
                fake_trapdoor = hashlib.sha1("".join(keyword).encode("utf-8")).hexdigest()
                temp_voc.append(fake_trapdoor)
                if known_queries.get(keyword):
                    temp_known[fake_trapdoor] = keyword
                eval_dict[fake_trapdoor] = keyword
            query_voc = temp_voc
            known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

            matchmaker = ScoreAttack(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc().numpy(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=query_voc,
                known_queries=known_queries,
            )

            for cluster_max_size in max_cluster_sizes:
                refinement_results = matchmaker.tf_predict_with_refinement(
                    list(eval_dict.keys()),
                    cluster_max_size=cluster_max_size,
                    ref_speed=10
                )

                refinement_accuracy = np.mean(
                    [eval_dict[td] in candidates for td, candidates in refinement_results.items()]
                )

                cluster_temp_sizes = [len(candidates) for candidates in refinement_results.values()]
                cluster_results[cluster_max_size]["cluster_sizes"].extend(cluster_temp_sizes)
                cluster_results[cluster_max_size]["accuracies"].append(refinement_accuracy)

        with open(result_file, "w", newline="") as csv_file:
            fieldnames = [
                "Cluster maximum size",
                "Mean",
                "Median",
                "q0.6",
                "q0.7",
                "q0.75",
                "q0.8",
                "q0.85",
                "q0.9",
                "q0.95",
                "q0.99",
                "Average acc",
                "Cluster sizes",
            ]
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()
            for cluster_max_size, results in cluster_results.items():
                writer.writerow(
                    {
                        "Cluster maximum size": cluster_max_size,
                        "Mean": np.mean(results["cluster_sizes"]),
                        "Median": np.quantile(results["cluster_sizes"], 0.5),
                        "q0.6": np.quantile(results["cluster_sizes"], 0.6),
                        "q0.7": np.quantile(results["cluster_sizes"], 0.7),
                        "q0.75": np.quantile(results["cluster_sizes"], 0.75),
                        "q0.8": np.quantile(results["cluster_sizes"], 0.8),
                        "q0.85": np.quantile(results["cluster_sizes"], 0.85),
                        "q0.9": np.quantile(results["cluster_sizes"], 0.9),
                        "q0.95": np.quantile(results["cluster_sizes"], 0.95),
                        "q0.99": np.quantile(results["cluster_sizes"], 0.99),
                        "Average acc": np.mean(results["accuracies"]),
                        "Cluster sizes": results["cluster_sizes"],
                    }
                )


def load_mails_example():
    logger.debug(f"Load enron dataset ...")
    occurrence_array, documents, keywords = load_enron(path_prefix='~/email_datasets/')

    logger.debug(f"Occurrence array shape: {occurrence_array.shape}")
    logger.debug(f"Documents shape: {documents.shape}")
    logger.debug(f"Keywords shape: {keywords.shape}")


def base_results(result_file=f"{kw_conjunction_size}-kws_base_attack-{start_time}.csv"):
    with tf.device("/device:CPU:0"):
        # voc_size_possibilities = [200]
        voc_size_possibilities = [500]
        known_queries_possibilities = [60, 30]
        experiment_params = [
            (i, j)
            for i in voc_size_possibilities         # Voc size
            for j in known_queries_possibilities    # known queries
            for _k in range(NB_REP)
        ]

        with open(result_file, "w", newline="") as csv_file:
            fieldnames = [
                "Nb similar docs",
                "Nb server docs",
                "Similar/Server voc size",
                "Nb queries seen",
                "Nb queries known",
                "Base acc",
                "Refinement acc",    # TODO: Move this code somewhere else
            ]
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()

            document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ = load_preprocessed_enron_float32(prefix='')

            # Split to N sparse tensors: one sparse tensor per email
            emails = tf.sparse.split(
                sp_input=document_keyword_occurrence,
                num_split=document_keyword_occurrence.dense_shape[0],
                axis=0,
            )

            logger.debug(f"Number of emails: {len(emails)}")

            modulus = len(known_queries_possibilities) * NB_REP

            keyword_combinations = None

            for (i, (voc_size, nb_known_queries)) in enumerate(experiment_params):
                logger.info(f"Experiment {i+1} out of {len(experiment_params)}")
                memory_usage()

                if i % modulus == 0:
                    logger.debug(f"Generate keyword combinations: nCr({voc_size}, {kw_conjunction_size})")
                    keyword_combinations = conjunctive_keyword_combinations_indices(
                        num_keywords=voc_size, kw_conjunction_size=kw_conjunction_size)

                # Split similar and real dataset
                email_ids = list(range(document_keyword_occurrence.dense_shape[0]))
                random.shuffle(email_ids)

                similar_doc_ids, stored_doc_ids = email_ids[:int(len(emails) * 0.4)], email_ids[int(len(emails) * 0.4):]
                logger.debug(f"Generated random email ids")

                similar_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in similar_doc_ids], axis=0)
                logger.debug(f"Similar docs shape: {similar_docs.dense_shape}")

                stored_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in stored_doc_ids], axis=0)
                logger.debug(f"Stored docs shape: {stored_docs.dense_shape}")

                # Extract keywords from similar dataset
                memory_usage()
                similar_extractor = ConjunctiveExtractor(
                    occurrence_array=similar_docs,
                    keyword_voc=sorted_keyword_voc,
                    keyword_occ=sorted_keyword_occ,
                    voc_size=voc_size,
                    kw_conjunction_size=kw_conjunction_size,
                    min_freq=1,
                    precalculated_artificial_keyword_combinations_indices=keyword_combinations,
                    multi_core=True,
                )
                nb_similar_docs = similar_extractor.occ_array.shape[1]
                memory_usage()

                # Extract keywords from real dataset
                real_extractor = ConjunctiveQueryResultExtractor(
                    stored_docs,
                    sorted_keyword_voc,
                    sorted_keyword_occ,
                    voc_size,
                    kw_conjunction_size,
                    1,
                    keyword_combinations,
                    True,
                )
                nb_server_docs = real_extractor.occ_array.shape[1]
                memory_usage()

                # Queries = 15% of artificial kws
                queryset_size = int(comb(voc_size, kw_conjunction_size, exact=True) * 0.15)

                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                # remove from memory not needed
                del real_extractor

                known_queries = generate_known_queries(
                    similar_wordlist=similar_extractor.get_sorted_voc().numpy(),
                    stored_wordlist=query_voc.numpy(),
                    nb_queries=nb_known_queries,
                )
                memory_usage()

                td_voc, known_queries, eval_dict = generate_trapdoors(
                    query_voc=query_voc.numpy(),
                    known_queries=known_queries,
                )
                # known_queries := Keys: Trapdoor tokens; Values: Keywords
                memory_usage()

                matchmaker = ConjunctiveScoreAttack(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc().numpy(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                )
                memory_usage()

                # Difference known queries and fake queries are the trapdoors we want to recover
                td_list = list(set(eval_dict.keys()).difference(matchmaker.known_queries.keys()))

                results = matchmaker.tf_predict(td_list, k=1)
                memory_usage()

                base_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                refinement_speed = int(0.05 * queryset_size)
                logger.info(f"Refinement speed: {refinement_speed}")

                # ALSO PRODUCE RESULTS WITH REFINEMENT
                # TODO: Move this code somewhere else
                # Prediction with refinement, but without clustering
                results = matchmaker.tf_predict_with_refinement(
                    td_list,
                    cluster_max_size=1,
                    ref_speed=refinement_speed
                )
                memory_usage()
                refinement_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                del similar_extractor

                writer.writerow(
                    {
                        "Nb similar docs": nb_similar_docs,
                        "Nb server docs": nb_server_docs,
                        "Similar/Server voc size": voc_size,
                        "Nb queries seen": queryset_size,
                        "Nb queries known": nb_known_queries,
                        "Base acc": base_accuracy,
                        "Refinement acc": refinement_accuracy,
                    }
                )

                # Flush, such that intermediate results don't get lost if something bad happens
                csv_file.flush()

                # Garbage collection to prevent high memory usage (tends to increase overtime)
                gc.collect()


def attack_comparison(result_file=f"{kw_conjunction_size}-kws_attack_comparison-{start_time}.csv"):
    with tf.device("/device:CPU:0"):
        # known queries
        experiment_params = [
            j for j in [5, 10, 20, 40] for _k in range(NB_REP)
        ]

        similar_voc_size = 21
        real_voc_size = 20
        with open(result_file, "w", newline="") as csv_file:
            fieldnames = [
                "Nb similar docs",
                "Nb server docs",
                "Similar voc size",
                "Server voc size",
                "Nb queries seen",
                "Nb queries known",
                "Base acc",
                # "Acc with cluster",
                "Acc with refinement",
            ]
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()

            document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ = load_preprocessed_dataset(prefix='')

            # Split to N sparse tensors: one sparse tensor per email
            emails = tf.sparse.split(
                sp_input=document_keyword_occurrence,
                num_split=document_keyword_occurrence.dense_shape[0],
                axis=0,
            )

            logger.debug(f"Number of emails: {len(emails)}")

            for (i, nb_known_queries) in enumerate(experiment_params):
                logger.info(f"Experiment {i+1} out of {len(experiment_params)}")

                # Split similar and real dataset
                email_ids = list(range(document_keyword_occurrence.dense_shape[0]))
                random.shuffle(email_ids)

                similar_doc_ids, stored_doc_ids = email_ids[:int(len(emails) * 0.4)], email_ids[int(len(emails) * 0.4):]
                logger.debug(f"Generated random email ids")

                similar_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in similar_doc_ids], axis=0)
                logger.debug(f"Similar docs shape: {similar_docs.dense_shape}")

                stored_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in stored_doc_ids], axis=0)
                logger.debug(f"Stored docs shape: {stored_docs.dense_shape}")

                # Queryset size fraction of real vocabulary size
                queryset_size = int(comb(real_voc_size, kw_conjunction_size, exact=True) * 0.15)

                # Extract keywords from similar dataset
                similar_extractor = ConjunctiveExtractor(
                    occurrence_array=similar_docs,
                    keyword_voc=sorted_keyword_voc,
                    keyword_occ=sorted_keyword_occ,
                    voc_size=similar_voc_size,
                    kw_conjunction_size=kw_conjunction_size,
                    min_freq=1,
                )

                # Extract keywords from real dataset
                real_extractor = ConjunctiveQueryResultExtractor(
                    stored_docs,
                    sorted_keyword_voc,
                    sorted_keyword_occ,
                    real_voc_size,
                    kw_conjunction_size,
                    1,
                )

                # Generate fake queries
                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                known_queries = generate_known_queries(
                    similar_wordlist=similar_extractor.get_sorted_voc().numpy(),
                    stored_wordlist=query_voc.numpy(),
                    nb_queries=nb_known_queries,
                )

                td_voc, known_queries, eval_dict = generate_trapdoors(
                    query_voc=query_voc.numpy(),
                    known_queries=known_queries,
                )
                # known_queries := Keys: Trapdoor tokens; Values: Keywords

                matchmaker = ConjunctiveScoreAttack(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc().numpy(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                )

                td_list = list(set(eval_dict.keys()).difference(matchmaker.known_queries.keys()))

                # Prediction for baseline, i.e. without clustering and refinement
                results = matchmaker.tf_predict(td_list, k=1)
                base_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                # Prediction with clustering
                # results = dict(matchmaker._sub_pred(0, td_list, cluster_max_size=10))
                # cluster_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                refinement_speed = int(0.05 * queryset_size)
                logger.info(f"Refinement speed: {refinement_speed}")

                # Prediction with clustering and refinement
                results = matchmaker.tf_predict_with_refinement(
                    td_list,
                    cluster_max_size=10,
                    ref_speed=refinement_speed
                )
                refinement_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                writer.writerow(
                    {
                        "Nb similar docs": similar_extractor.occ_array.shape[0],
                        "Nb server docs": real_extractor.occ_array.shape[0],
                        "Similar voc size": similar_voc_size,
                        "Server voc size": real_voc_size,
                        "Nb queries seen": queryset_size,
                        "Nb queries known": nb_known_queries,
                        "Base acc": base_accuracy,
                        # "Acc with cluster": cluster_accuracy,
                        "Acc with refinement": refinement_accuracy,
                    }
                )

                # Flush, such that intermediate results don't get lost if something bad happens
                csv_file.flush()


def known_data(result_file=f"{kw_conjunction_size}-kws_known_data-{start_time}.csv"):
    with tf.device("/device:CPU:0"):
        # voc_size_possibilities = [200]
        known_data_percentages = [0.7]
        voc_size_possibilities = [500]
        known_queries_possibilities = [10]
        experiment_params = [
            (h, i, j)
            for h in known_data_percentages
            for i in voc_size_possibilities  # Voc size
            for j in known_queries_possibilities  # known queries
            for _k in range(NB_REP)
        ]

        with open(result_file, "w", newline="") as csv_file:
            fieldnames = [
                "Percentage known",
                "Nb server docs",
                "Similar/Server voc size",
                "Nb queries seen",
                "Nb queries known",
                "Base acc",
                "Refinement acc",  # TODO: Move this code somewhere else
            ]
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()

            document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ = load_preprocessed_enron_float32(prefix='')

            # Split to N sparse tensors: one sparse tensor per email
            emails = tf.sparse.split(
                sp_input=document_keyword_occurrence,
                num_split=document_keyword_occurrence.dense_shape[0],
                axis=0,
            )

            logger.debug(f"Number of emails: {len(emails)}")

            keyword_combinations = None
            memory_usage()
            for (i, (known_frac, voc_size, nb_known_queries)) in enumerate(experiment_params):
                logger.info(f"Experiment {i + 1} out of {len(experiment_params)}")
                tf.keras.backend.clear_session()
                gc.collect()

                memory_usage()
                # Split similar and real dataset
                email_ids = list(range(document_keyword_occurrence.dense_shape[0]))
                random.shuffle(email_ids)

                known_doc_ids, stored_doc_ids = email_ids[:int(len(emails) * known_frac)], email_ids
                logger.debug(f"Generated random email ids")

                known_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in known_doc_ids], axis=0)
                logger.debug(f"Known docs shape: {known_docs.dense_shape}")

                stored_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in stored_doc_ids], axis=0)
                logger.debug(f"Stored docs shape: {stored_docs.dense_shape}")
                memory_usage()

                # Extract keywords from similar dataset
                known_extractor = ConjunctiveExtractor(
                    occurrence_array=known_docs,
                    keyword_voc=sorted_keyword_voc,
                    keyword_occ=sorted_keyword_occ,
                    voc_size=voc_size,
                    kw_conjunction_size=kw_conjunction_size,
                    min_freq=1,
                    precalculated_artificial_keyword_combinations_indices=keyword_combinations,
                    multi_core=True,
                )
                memory_usage()

                # with tf.device("/device:CPU:0"):
                # Extract keywords from real dataset
                real_extractor = ConjunctiveQueryResultExtractor(
                    stored_docs,
                    sorted_keyword_voc,
                    sorted_keyword_occ,
                    voc_size,
                    kw_conjunction_size,
                    1,
                    keyword_combinations,
                    True,
                )
                memory_usage()
                # Queries = 15% of artificial kws
                queryset_size = int(comb(voc_size, kw_conjunction_size, exact=True) * 0.15)

                logger.debug(f"generate fake queries: {queryset_size}")
                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                logger.debug(f"query_array.shape: {tf.shape(query_array)}")

                known_queries = generate_known_queries(
                    similar_wordlist=known_extractor.get_sorted_voc().numpy(),
                    stored_wordlist=query_voc.numpy(),
                    nb_queries=nb_known_queries,
                )

                logger.debug(f"query_array.shape: {tf.shape(query_array)}")

                nb_real_docs = real_extractor.occ_array.shape[1]

                td_voc, known_queries, eval_dict = generate_trapdoors(
                    query_voc=query_voc.numpy(),
                    known_queries=known_queries,
                )
                # known_queries := Keys: Trapdoor tokens; Values: Keywords
                memory_usage()

                logger.debug(f"query_array.shape: {tf.shape(query_array)}")

                # with tf.device("/device:CPU:0"):
                matchmaker = ConjunctiveScoreAttack(
                    keyword_occ_array=known_extractor.occ_array,
                    keyword_sorted_voc=known_extractor.get_sorted_voc().numpy(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                )
                memory_usage()
                # Difference known queries and fake queries are the trapdoors we want to recover
                td_list = list(set(eval_dict.keys()).difference(matchmaker.known_queries.keys()))

                results = matchmaker.tf_predict(td_list, k=1)
                base_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                # with tf.device("/device:CPU:0"):
                refinement_speed = int(0.05 * queryset_size)
                logger.info(f"Refinement speed: {refinement_speed}")

                # ALSO PRODUCE RESULTS WITH REFINEMENT
                # TODO: Move this code somewhere else
                # Prediction with refinement, but without clustering
                memory_usage()
                results = matchmaker.tf_predict_with_refinement(
                    td_list,
                    cluster_max_size=1,
                    ref_speed=refinement_speed
                )
                memory_usage()
                refinement_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])
                memory_usage()

                writer.writerow(
                    {
                        "Percentage known": known_frac,
                        "Nb server docs": nb_real_docs,
                        "Similar/Server voc size": voc_size,
                        "Nb queries seen": queryset_size,
                        "Nb queries known": nb_known_queries,
                        "Base acc": base_accuracy,
                        "Refinement acc": refinement_accuracy,
                    }
                )

                # Flush, such that intermediate results don't get lost if something bad happens
                csv_file.flush()

                del known_extractor
                del real_extractor
                del matchmaker

                memory_usage()
                # Garbage collection
                gc.collect()

                memory_usage()


def apache_reduced():
    ratio = 30109 / 50878
    apache_full = extract_apache_ml()
    apache_red, _ = split_df(apache_full, ratio)
    return apache_red


def document_set_results(result_file=f"{kw_conjunction_size}-kws_document_set-{start_time}.csv"):
    with tf.device("/device:CPU:0"):
        email_extractors = [
            (extract_sent_mail_contents, "Enron"),
            (extract_apache_ml, "Apache"),
            (apache_reduced, "Apache reduced"),
        ]
        queryset_sizes = [i for i in [150, 300, 600, 1000] for _j in range(NB_REP)]

        similar_voc_size = 46
        real_voc_size = 46
        with open(result_file, "w", newline="") as csv_file:
            fieldnames = [
                "Dataset",
                "Nb similar docs",
                "Nb server docs",
                "Similar voc size",
                "Server voc size",
                "Nb queries seen",
                "Nb queries known",
                "Acc",
            ]
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()
            i = 0
            for extractor, dataset in email_extractors:
                emails = extractor()
                for queryset_size in queryset_sizes:
                    i += 1
                    logger.info(f"Experiment {i} out of {len(email_extractors)*len(queryset_sizes)}")

                    # Split similar and real dataset
                    similar_docs, stored_docs = split_df(d_frame=emails, frac=0.4)

                    # Extract keywords from similar dataset
                    similar_extractor = KeywordExtractor(
                        similar_docs,
                        similar_voc_size,
                        min_freq=1,
                        with_kw_conjunction_size=kw_conjunction_size
                    )

                    # Extract keywords from real dataset
                    real_extractor = QueryResultExtractor(
                        stored_docs,
                        real_voc_size,
                        1,
                        with_artificial_kw_size=kw_conjunction_size
                    )

                    query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                    known_queries = old_generate_known_queries(
                        similar_wordlist=similar_extractor.get_sorted_voc().numpy(),
                        stored_wordlist=query_voc,
                        nb_queries=15,
                    )

                    td_voc = []
                    temp_known = {}
                    eval_dict = {}  # Keys: Trapdoor tokens; Values: Keywords
                    for keyword in query_voc:
                        fake_trapdoor = hashlib.sha1("".join(keyword).encode("utf-8")).hexdigest()
                        td_voc.append(fake_trapdoor)
                        if known_queries.get(keyword):
                            temp_known[fake_trapdoor] = keyword
                        eval_dict[fake_trapdoor] = keyword
                    known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

                    matchmaker = ScoreAttack(
                        keyword_occ_array=similar_extractor.occ_array,
                        keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                        trapdoor_occ_array=query_array,
                        trapdoor_sorted_voc=td_voc,
                        known_queries=known_queries,
                    )

                    td_list = list(set(eval_dict.keys()).difference(matchmaker.known_queries.keys()))

                    refinement_speed = int(0.05 * queryset_size)

                    results = matchmaker.predict_with_refinement(td_list, cluster_max_size=10, ref_speed=refinement_speed)
                    refinement_accuracy = np.mean([eval_dict[td] in candidates for td, candidates in results.items()])

                    writer.writerow(
                        {
                            "Dataset": dataset,
                            "Nb similar docs": similar_extractor.occ_array.shape[0],
                            "Nb server docs": real_extractor.occ_array.shape[0],
                            "Similar voc size": similar_voc_size,
                            "Server voc size": real_voc_size,
                            "Nb queries seen": queryset_size,
                            "Nb queries known": 15,
                            "Acc": refinement_accuracy,
                        }
                    )

                    # Flush, such that intermediate results don't get lost if something bad happens
                    csv_file.flush()
