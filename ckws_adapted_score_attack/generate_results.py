"""Script to reproduce all the results presented in the QueRyvolution paper.
"""
import os
import colorlog
import logging
import tensorflow as tf
from ckws_adapted_score_attack.src.config import MAX_CPUS

# number of threads used for parallelism between independent operations:
tf.config.threading.set_inter_op_parallelism_threads(MAX_CPUS)

# number of threads used within an individual operation for parallelism
tf.config.threading.set_intra_op_parallelism_threads(MAX_CPUS)

# run operation on CPU if not implemented for GPU
tf.config.set_soft_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Could not set_memory_growth True: {e}")

from ckws_adapted_score_attack.src.common import setup_logger
from ckws_adapted_score_attack.src.result_procedures import (
    understand_variance,
    cluster_size_statistics,
    base_results,
    load_mails_example,
    attack_comparison,
    document_set_results,
    known_data,
)

logger = colorlog.getLogger("CKWS-Adapted-Refined-Score-Attack")

PROFILER_ENABLED = False

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


if __name__ == "__main__":
    setup_logger()
    FORMATTER = logging.Formatter("[%(asctime)s %(levelname)s] %(module)s: %(message)s")
    file_handler = logging.FileHandler("results_known_2_p70_10.log")
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)

    procedures = (
        # load_mails_example,
        base_results,
        # attack_comparison,
        # cluster_size_statistics,
        # understand_variance,
        # document_set_results,
        # known_data,
    )

    # tf.debugging.set_log_device_placement(True)
    if PROFILER_ENABLED:
        tf.profiler.experimental.start('profiler/log')

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 2
    # config.gpu_options.allow_growth = True
    # session = tf.compat.v1.InteractiveSession(config=config)

    cpus = tf.config.list_physical_devices('CPU')
    logical_cpus = tf.config.list_logical_devices('CPU')
    logger.info(f"cpus ({len(cpus)}): {cpus}; logical ({len(logical_cpus)}): {logical_cpus}")

    # logger.debug(f"get_memory_growth(gpus[0]): {tf.config.experimental.get_memory_growth(gpus[0])}")

    for procedure in procedures:
        logger.info(f"Starting procedure {procedure.__name__}")
        procedure()
        logger.info(f"Procedure {procedure.__name__} ended")

    if PROFILER_ENABLED:
        tf.profiler.experimental.stop()
