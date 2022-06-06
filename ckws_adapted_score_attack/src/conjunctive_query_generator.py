import math
import random

from typing import List, Tuple

import colorlog
import numpy as np
import tensorflow as tf
import scipy.stats as stats

from ckws_adapted_score_attack.src.conjunctive_extraction import ConjunctiveExtractor

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")


class ConjunctiveQueryResultExtractor(ConjunctiveExtractor):
    """Just a keyword extractor augmented with a query generator.
    It corresponds to the index in the server. The fake queries are
    seen by the attacker.
    """

    def __init__(self, *args, distribution="uniform", **kwargs):
        super().__init__(*args, **kwargs)
        n = len(self.sorted_voc)
        x = np.arange(1, n + 1)
        if distribution == "uniform":
            weights = np.ones(n) / n
        elif distribution == "zipfian":
            a = 1.0
            weights = x ** (-a)
            weights /= weights.sum()
        elif distribution == "inv_zipfian":
            a = 1.0
            weights = (n - x + 1) ** (-a)
            weights /= weights.sum()
        else:
            raise ValueError("Distribution not supported.")
        self._rv = stats.rv_discrete(name="query_distribution", values=(x, weights))

    def __str__(self):
        return "Basic"

    def _generate_random_sample(self, size=1) -> list:
        """Generate a sample with unique element thanks to the random variable
        define in the __init__.

        Keyword Arguments:
            size {int} -- Size of the sample of random queries (default: {1})

        Returns:
            sample_list -- List of indices picked at random
        """
        sample_set = set(self._rv.rvs(size=size) - 1)
        queries_remaining = size - len(sample_set)
        while queries_remaining > 0:
            # The process is repeated until having the correct size.
            # It is not elegant, but in IKK and Cash, they present
            # queryset with unique queries.
            sample_set = sample_set.union(self._rv.rvs(size=queries_remaining) - 1)
            queries_remaining = size - len(sample_set)
        sample_list = list(sample_set)  # Cast needed to index np.array

        return sample_list

    def get_fake_queries(self, size=1, hide_nb_files=True) -> Tuple[tf.Tensor, tf.Tensor]:
        logger.info("Generating fake queries")
        sample_list = self._generate_random_sample(size=size)

        # sample_list is sorted since a set of integers is sorted
        logger.debug(f"len sample_list: {len(sample_list)} ({len(set(sample_list))})")

        # [self.sorted_voc[ind] for ind in sample_list]
        query_voc = tf.gather(self.sorted_voc, sample_list, name="fake_queries_gather_sorted_voc")

        # query_arr = self.occ_array[:, sample_list]
        query_arr = tf.gather(self.occ_array, sample_list, axis=0, name="fake_queries_gather_occ_array")
        # i-th element of column is 0/1 depending if the i-th document includes the keyword

        if hide_nb_files:
            query_arr = tf.gather(query_arr, tf.squeeze(tf.where(tf.reduce_sum(query_arr, axis=0) != 0)), axis=1)

            # We remove every line containing only zeros, so we hide the nb of documents stored
            # i.e. we remove every documents not involved in the queries
            # query_arr = query_arr[~np.all(query_arr == 0, axis=1)]
        return query_arr, query_voc
