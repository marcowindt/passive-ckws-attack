import argparse

import colorlog

import tensorflow as tf
import random

from scipy.special import comb

from ckws_adapted_score_attack.src.conjunctive_extraction import ConjunctiveExtractor, generate_known_queries, \
    generate_trapdoors
from ckws_adapted_score_attack.src.conjunctive_matchmaker import ConjunctiveScoreAttack
from ckws_adapted_score_attack.src.conjunctive_query_generator import ConjunctiveQueryResultExtractor
from ckws_adapted_score_attack.src.common import setup_logger
from ckws_adapted_score_attack.src.email_extraction import extract_sent_mail_contents, extract_apache_ml, \
    load_preprocessed_dataset, load_preprocessed_enron_float32

from ckws_adapted_score_attack.src.trapdoor_matchmaker import memory_usage

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")

DocumentSetExtraction = {
    "enron": extract_sent_mail_contents,
    "apache": extract_apache_ml,
    "enron_preprocessed": load_preprocessed_dataset,
    "enron_preprocessed_float32": load_preprocessed_enron_float32,
}

MULTI_CORE = True


def attack_procedure(*args, **kwargs):
    """Procedure to simulate an inference attack.
    """
    setup_logger()
    logger.debug(f"tf.version: {tf.version.VERSION} ({tf.version.GRAPH_DEF_VERSION})")
    # Params
    similar_voc_size = kwargs.get("similar_voc_size", 1000)
    server_voc_size = kwargs.get("server_voc_size", 1000)

    queryset_size = kwargs.get("queryset_size", int(0.15 * server_voc_size))
    nb_known_queries = kwargs.get("nb_known_queries", int(queryset_size * 0.15))

    attack_dataset = kwargs.get("attack_dataset", "enron_preprocessed_float32")

    kw_conjunction_size = kwargs.get('kw_conjunction_size', 2)

    logger.debug(f"Server vocabulary size: {server_voc_size}")
    logger.debug(f"Similar vocabulary size: {similar_voc_size}")
    logger.debug(f"Artificial keyword size: {kw_conjunction_size}")
    logger.debug(f"Query set size: {queryset_size}")
    logger.debug(f"Nb known queries: {nb_known_queries}")

    if kwargs.get("L2"):
        logger.debug("L2 Scheme")
    else:
        logger.debug(f"L1 Scheme => Queryset size: {queryset_size}, Known queries: {nb_known_queries}")

    try:
        extraction_procedure = DocumentSetExtraction[attack_dataset]
    except KeyError:
        raise ValueError("Unknown dataset")

    cpus = tf.config.list_physical_devices('CPU')

    logger.info(f"cpus ({len(cpus)}): {cpus}")

    logger.info("ATTACK BEGINS")

    deterministic_split = False
    memory_usage()
    document_keyword_occurrence, sorted_keyword_voc, sorted_keyword_occ = extraction_procedure(prefix='.')
    memory_usage()
    logger.debug(f"Split in 0.4 similar and 0.6 real")

    # Split to N sparse tensors: one sparse tensor per email
    emails = tf.sparse.split(
        sp_input=document_keyword_occurrence,
        num_split=document_keyword_occurrence.dense_shape[0],
        axis=0
    )

    logger.debug(f"emails len: {len(emails)}")
    memory_usage()
    email_ids = list(range(document_keyword_occurrence.dense_shape[0]))
    if not deterministic_split:
        random.shuffle(email_ids)

    similar_doc_ids, stored_doc_ids = email_ids[:int(len(emails) * 0.4)], email_ids[int(len(emails) * 0.4):]
    logger.debug(f"Generated random email ids")

    similar_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in similar_doc_ids], axis=0)
    logger.debug(f"Similar docs shape: {similar_docs.dense_shape}")

    stored_docs = tf.sparse.concat(sp_inputs=[emails[i] for i in stored_doc_ids], axis=0)
    logger.debug(f"Stored docs shape: {stored_docs.dense_shape}")
    memory_usage()
    # KEYWORD EXTRACTION
    logger.info("Extracting keywords from similar documents.")
    similar_extractor = ConjunctiveExtractor(
        occurrence_array=similar_docs,
        keyword_voc=sorted_keyword_voc,
        keyword_occ=sorted_keyword_occ,
        voc_size=similar_voc_size,
        kw_conjunction_size=kw_conjunction_size,
        min_freq=1,
        precalculated_artificial_keyword_combinations_indices=None,
        multi_core=MULTI_CORE,
    )
    memory_usage()
    logger.info("Extracting keywords from stored documents.")
    size_real_stored_docs = stored_docs.dense_shape[0]
    real_extractor = ConjunctiveQueryResultExtractor(
        stored_docs,
        sorted_keyword_voc,
        sorted_keyword_occ,
        server_voc_size,
        kw_conjunction_size,
        1,
        None,
        MULTI_CORE,
    )
    memory_usage()
    # QUERY GENERATION
    logger.info(f"Generating {queryset_size} queries from stored documents #docs={size_real_stored_docs}")
    query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

    # logger.debug(f"query_array: {query_array}")
    # logger.debug(f"query_voc  : {query_voc}")

    del real_extractor  # Reduce memory usage especially when applying countermeasures
    memory_usage()
    logger.debug(f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)")
    known_queries = generate_known_queries(
        # Extracted with uniform law
        similar_wordlist=similar_extractor.get_sorted_voc().numpy(),
        stored_wordlist=query_voc.numpy(),
        nb_queries=nb_known_queries,
    )
    memory_usage()
    logger.debug("Hashing the keywords of the stored documents (transforming them into trapdoor tokens)")

    # Trapdoor token => convey no information about the corresponding keyword
    query_voc, known_queries, eval_dict = generate_trapdoors(
        query_voc=query_voc.numpy(),
        known_queries=known_queries,
    )
    memory_usage()
    # tmp_sorted_voc = similar_extractor.get_sorted_voc()
    # logger.info(f"Words in the sorted vocabulary {tmp_sorted_voc}")

    # logger.debug(f"Query vocabulary {query_voc}")
    # logger.debug(f"Known queries {known_queries}")

    # THE INFERENCE ATTACK
    matchmaker = ConjunctiveScoreAttack(
        keyword_occ_array=similar_extractor.occ_array,
        keyword_sorted_voc=similar_extractor.get_sorted_voc().numpy(),
        trapdoor_occ_array=query_array,
        trapdoor_sorted_voc=query_voc,
        known_queries=known_queries,
    )
    memory_usage()
    del similar_extractor
    memory_usage()
    base_acc = matchmaker.base_accuracy(k=1, eval_dico=eval_dict)[0]                    # Baseline
    memory_usage()
    # ref_acc_1 = matchmaker.queryvolution_accuracy(eval_dico=eval_dict)[0]               # With clustering
    # memory_usage()
    ref_acc = matchmaker.queryvolution_accuracy_no_clustering(eval_dico=eval_dict)[0]   # Without clustering
    memory_usage()
    logger.info(
        f"Base accuracy: {base_acc} / Refinement accuracy: {ref_acc} "
    )
    # NB: To be sure there is no bias in the algorithm we can compute the accuracy manually
    # as it is done for the refinement accuracy here.
    # The matchmaker and the eval_dict are returned, so you
    # can run your own test in a Python terminal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--similar-voc-size",
        type=int,
        default=50,
        help="Size of the vocabulary extracted from similar documents.",
    )
    parser.add_argument(
        "--server-voc-size",
        type=int,
        default=50,
        help="Size of the vocabulary stored in the server.",
    )
    parser.add_argument(
        "--queryset-size",
        type=int,
        default=int(comb(50, 2, exact=True) * 0.15),
        help="Number of queries which have been observed.",
    )
    parser.add_argument(
        "--nb-known-queries",
        type=int,
        default=10,
        help="Number of queries known by the attacker. Known Query=(Trapdoor, Corresponding Keyword)",
    )
    parser.add_argument(
        "--L2",
        default=True,
        action="store_true",
        help="Whether the server has an L2 scheme or not",
    )
    parser.add_argument(
        "--attack-dataset",
        type=str,
        default="enron_preprocessed_float32",
        help="Dataset used for the attack",
    )
    parser.add_argument(
        "--kw-conjunction-size",
        default=2,
        type=int,
        help="The fixed number of keywords in one conjunctive keyword query.",
    )

    params = parser.parse_args()
    assert (0 < params.nb_known_queries <= params.queryset_size)

    # tf.profiler.experimental.start('profiler/log')
    attack_procedure(**vars(params))
    # tf.profiler.experimental.stop()
