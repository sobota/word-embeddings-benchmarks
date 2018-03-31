import datetime
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from .feature_view import learn_logit_reg, generate_figure, store_data, process_CSLB
from web.embeddings import Embedding


def gaussian_embedding(dim, words, loc, scale):
    """
    Generate embedding based on given words, with vectors from gaussian distribution.
    :param words: for each word gaussian vector is generated
    :return: dict with vectors from gaussian distributions
    """
    return {w: np.random.normal(loc=loc, scale=scale, size=dim) for w in words}


def cslb_on_gaussian_experiment(dimension, loc=0.0, scale=1.0, cslb_path='./CSLB',
                                save_path='./', figure_desc='', n_jobs=4, random_state=0, nb_hyper=20, max_iter=1800,
                                show_dialog=False):
    """
    Run experiment on syntactical gaussian noise embedding. Is analogy to feature_view experiment.

    Based on whitepaper:
    'Are distributional representations ready for the real world? Evaluating word vectors for grounded perceptual meaning'
    :param dimension: length of vectors in embedding
    :param loc: mean of gaussian distribution
    :param scale: standard deviation of gaussian distribution
    :param cslb_path: to folder with cslb files norms.dat and feature_matrix.dat
    :param n_jobs: numbers of threads used for learning
    :param save_path: path where figures and progress files will be saved
    :param figure_desc: description after 'CSLB' in title
    :param random_state: seed. Important for replicability
    :param nb_hyper: number of hyperparm value to select
    :param max_iter: max iter of SGD
    :param show_dialog: when is False matplotlib Agg backend is used for generating figures
    """
    cslb_matrix = os.path.join(cslb_path, 'feature_matrix.dat')
    cslb_norm = os.path.join(cslb_path, 'norms.dat')

    if not os.path.isfile(cslb_matrix):
        print("Download CSLB file first from website: http://www.csl.psychol.cam.ac.uk/propertynorms/", file=sys.stderr)
        raise FileNotFoundError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), cslb_matrix)

    if not os.path.isfile(cslb_norm):
        print("Download CSLB file first from website: http://www.csl.psychol.cam.ac.uk/propertynorms/", file=sys.stderr)
        raise FileNotFoundError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), cslb_norm)

    df = pd.read_csv(cslb_matrix, sep='\t', index_col=0)
    words = df.index.values.astype(str)

    embedding = Embedding.from_dict(
        gaussian_embedding(dim=dimension, words=words, loc=loc, scale=scale))

    cleaned = process_CSLB(embedding, feature_matrix_path=cslb_matrix)

    logging.info('Shape of cleaned CSLB is {}'.format(cleaned.shape))
    cdf = cleaned.transpose()

    concepts = [str(x) for x in cdf.index]
    features = cdf.columns

    fs_id, fp_id = learn_logit_reg(embedding=embedding, features=features, concepts=concepts, cleaned_norms=cleaned,
                                   n_jobs=n_jobs, random_state=random_state, max_iter=max_iter, nb_hyper=nb_hyper)
    logging.info('Generating Plots')

    now_dt = datetime.datetime.now()
    fig_path = os.path.join(save_path,
                            'cslb_{}_{:%d-%m-%Y_%H:%M}.png'.format(re.sub('\s', '', figure_desc), now_dt))

    generate_figure(fs_id, fig_title=figure_desc, norms_path=cslb_norm, fig_path=fig_path, show_visual=show_dialog)

    store_path = os.path.join(save_path,
                              'cslb_f1_params_{}_{:%d-%m-%Y_%H:%M}.csv'.format(re.sub('\s', '', figure_desc), now_dt))
    store_data(fs_id, fp_id, store_path)
