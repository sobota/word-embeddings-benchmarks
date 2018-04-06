from __future__ import print_function
import datetime
import logging
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from tqdm import tqdm


def _process_CSLB(feature_matrix_path, *embs):
    df = pd.read_csv(feature_matrix_path, sep='\t', index_col=0)

    t = df.transpose()
    t.iloc[t.values > 1] = 1

    iv = df.index.values.astype(str)

    word_to_rm = set([])

    # filter double meaning words
    for cw in iv:
        sw = str(cw)

        ssw = sw.split('_(')
        if len(ssw) >= 2:
            word_to_rm.add(sw)
        else:
            # remove two-word names according to paper
            # 'guinea_pig', 'rolls_royce'
            ssw = sw.split('_')
            if len(ssw) >= 2:
                word_to_rm.add(sw)

    word_to_rm = [str(x) for x in word_to_rm]

    # drop words non aviable in embeddings corpora
    concepts_str = [str(x) for x in t.columns]

    def get_unexisting_words(emb):
        lack_words = []

        for e in emb:
            for c in concepts_str:
                if c not in e.words:
                    lack_words.append(c)

        return set(lack_words)

    uw = get_unexisting_words(embs)
    word_to_rm.extend(uw)

    # remove before feature selection
    cleaned_df = t.drop(word_to_rm, axis=1)

    f_summed = cleaned_df.sum(axis=1)

    # feature with 5 or more positive examples
    selected_feature = f_summed.loc[f_summed >= 5]

    # DataFrame with removed absent word in embedding and
    cleaned_df = cleaned_df.loc[selected_feature.index.values.astype(str)]

    return cleaned_df


def _learn_logit_reg(embedding, features, concepts, cleaned_norms, n_jobs=4, random_state=None, nb_hyper=20,
                     max_iter=None):
    X = np.asarray([embedding[c_w] for c_w in concepts])

    # features in semantic norms
    def train(f):
        y = cleaned_norms.loc[f].values

        gs = GridSearchCV(
            estimator=SGDClassifier(loss='log', class_weight='balanced', eta0=0.01, learning_rate='optimal',
                                    n_jobs=n_jobs, max_iter=max_iter, tol=1e-3, random_state=random_state),
            cv=LeaveOneOut(), param_grid={'alpha': np.linspace(start=0.0001, stop=16.0, num=nb_hyper)}, n_jobs=n_jobs,
            scoring=make_scorer(f1_score))

        gs.fit(X, y)
        f1s = f1_score(y_true=y, y_pred=gs.predict(X))
        cv_parms = gs.best_params_

        logging.info('F1 score {} for feature {} with params {}'.format(f1s, f, cv_parms))

        end_coef = gs.best_estimator_.coef_
        end_iter = gs.best_estimator_.n_iter_
        end_intercept = gs.best_estimator_.intercept_

        params = [cv_parms['alpha'], end_iter, end_intercept[0]]
        params.extend(end_coef.flatten())

        return f, f1s, params

    data = map(train, tqdm(features, desc='Train logreg for each feature'))

    data = np.asarray(list(data))

    features_f1_scored = dict(zip(data[:, 0], data[:, 1]))
    features_params = np.delete(data, np.s_[1], axis=1)

    return features_f1_scored, features_params


def _generate_figure(fs_id, fig_title, norms_path='./CSLB',
                     fig_path='./cslb_feature_view_{:%d-%m-%Y_%H:%M}.png'.format(datetime.datetime.now()),
                     show_visual=False):
    """
    This function generate picture for feature categories according to CLSB classification

    Parameters
    ----------
    fs_id: dict
        Keys are features and F1-score are values for logreg trained for it

    norms_path: string
        Path to cslb dat files

    fig_path: string
    Path where figure is saved. If is None figure is not stored

    fig_title: string
        added to title after CSLB

    show_visual: bool
        show swarmplot using matplotlib
    """

    # for feature fit
    categ = pd.read_csv(norms_path, sep='\t')
    categ = categ[['feature type', 'concept', 'feature']]

    def get_feature_category(word_str):
        """Get all feature category"""
        word_str = word_str.replace('_', ' ')
        return categ.loc[categ['feature'] == word_str]['feature type'].unique().astype(str)

    def binsort_features(feature_f1):
        """Partition by feature type eg. 'encycopedic'"""
        cat_val = defaultdict(list)
        for f in feature_f1.keys():
            for c in get_feature_category(f):
                cat_val[c].append(feature_f1[f])

        return cat_val

    bf = binsort_features(fs_id)

    df = pd.DataFrame.from_dict(bf, orient='index')

    logging.info('Current matplotlib backend is {}'.format(plt.get_backend()))

    # workaround for running matplotlib image generation without Xserver
    if not show_visual:
        logging.info('Switching matplotlib backend to Agg')
        plt.switch_backend('agg')

    sns.set_style("whitegrid")

    tdf = df.transpose()
    tdf.sort_index(axis=1, inplace=True)

    sns.boxplot(data=tdf, boxprops=dict(alpha=0.2))
    ax = sns.swarmplot(data=tdf)
    plt.title('CSLB {}'.format(fig_title))

    ax.set(xlabel='Feature category', ylabel='Feature fit score')

    if fig_path is not None:
        ax.get_figure().savefig(fig_path)
        logging.info('Figure saved: {}'.format(fig_path))

    if show_visual:
        plt.show()

    plt.close()


def _store_data(fs_id, fp_id, file_path):
    score_df = pd.DataFrame(list(fs_id.items()), columns=['Feature', 'F1score'])

    params_data = {'Feature': fp_id[:, 0], 'Alpha': fp_id[:, 1], 'EndIter': fp_id[:, 2], 'Intercept': fp_id[:, 3],
                   'Coefs': list(fp_id[:, 4:])}

    params_df = pd.DataFrame(params_data, columns=['Feature', 'Alpha', 'EndIter', 'Intercept', 'Coefs'])

    df = score_df.merge(right=params_df, on='Feature')

    df = df.set_index('Feature')
    df.sort_index(inplace=True)
    logging.info('Saving: {}'.format(file_path))
    df.to_csv(file_path)


def generate_figure(path, fig_title):
    df = pd.read_csv(path, index_col=0)
    d = dict(zip(df.index.tolist, df.F1score.values))

    _generate_figure(fs_id=d, fig_title=fig_title)


def evaluate_cslb(embedding, cslb_path='./CSLB',
                  save_path='./', figure_desc='', n_jobs=4, random_state=0, nb_hyper=20, max_iter=1800,
                  display_figure=False):
    """
    Evaluate how well embedding encode features perceptual features. This experiment use CSLB semantic norms.

    Parameters
    ----------
    embedding: Embedding object
        Loaded embedding

    cslb_path: string
        Path to folder with cslb files norms.dat and feature_matrix.dat

    n_jobs: int
        Numbers of threads used for learning

    save_path: string
        Path where figures and progress files will be saved

    figure_desc: string
        Description after 'CSLB' in title

    random_state: int or RandomState
        Seed important for replicability

    nb_hyper: int
        Number of hyperparm value to select. Used by logistics regression for each feature independly.

    max_iter: int
        Max iter of SGD

    display_figure: bool
        When is False matplotlib Agg backend is used for generating figures

    References
    ----------
        Reference paper: 'Are distributional representations ready for the real world? Evaluating word vectors for grounded perceptual meaning'
        https://arxiv.org/abs/1705.11168
    """
    cslb_matrix = os.path.join(cslb_path, 'feature_matrix.dat')
    cslb_norm = os.path.join(cslb_path, 'norms.dat')

    if not os.path.isfile(cslb_matrix):
        print("Download CSLB file first from website: http://www.csl.psychol.cam.ac.uk/propertynorms/", file=sys.stderr)
        raise FileNotFoundError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), cslb_matrix)

    if not os.path.isfile(cslb_norm):
        print("Download CSLB file first from website: http://www.csl.psychol.cam.ac.uk/propertynorms/", file=sys.stderr)
        raise FileNotFoundError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), cslb_norm)

    cleaned = _process_CSLB(cslb_matrix, embedding)

    logging.info('Shape of cleaned CSLB is {}'.format(cleaned.shape))
    cdf = cleaned.transpose()

    concepts = [str(x) for x in cdf.index]
    features = cdf.columns

    fs_id, fp_id = _learn_logit_reg(embedding=embedding, features=features, concepts=concepts, cleaned_norms=cleaned,
                                    n_jobs=n_jobs, random_state=random_state, max_iter=max_iter, nb_hyper=nb_hyper)
    logging.info('Generating Plots & Storing Data')

    now_dt = datetime.datetime.now()

    fig_path = os.path.join(save_path,
                            'cslb_{}_{:%d-%m-%Y_%H:%M}.png'.format(re.sub('\s', '', figure_desc), now_dt))
    store_path = os.path.join(save_path,
                              'cslb_f1_params_{}_{:%d-%m-%Y_%H:%M}.csv'.format(re.sub('\s', '', figure_desc), now_dt))

    _store_data(fs_id, fp_id, store_path)

    _generate_figure(fs_id, fig_title=figure_desc, norms_path=cslb_norm, fig_path=fig_path, show_visual=display_figure)
