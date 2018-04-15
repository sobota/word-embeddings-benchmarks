from __future__ import print_function
import datetime
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from tqdm import tqdm


def process_CSLB(feature_matrix_path, *embs):
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
            cv=LeaveOneOut(), param_grid={'alpha': np.logspace(start=-7, stop=1.0, num=nb_hyper)}, n_jobs=n_jobs,
            scoring=make_scorer(f1_score))

        gs.fit(X, y)
        f1s = f1_score(y_true=y, y_pred=gs.predict(X))
        cv_parms = gs.best_params_

        logging.info('F1 score {} for feature {} with params {}'.format(f1s, f, cv_parms))

        end_coef = gs.best_estimator_.coef_
        end_iter = gs.best_estimator_.n_iter_
        end_intercept = gs.best_estimator_.intercept_

        # f, f1score, alpha, end_iter, intercept, rest_of_params
        params = [f, f1s, cv_parms['alpha'], end_iter, end_intercept[0]]
        params.extend(end_coef.flatten())

        return params

    data = list(map(train, tqdm(features, desc='Train logreg for each feature')))

    data = np.asarray(data)

    features_f1_scored = dict(zip(data[:, 0], data[:, 1]))
    features_params = np.delete(data, np.s_[1], axis=1)

    return features_f1_scored, features_params


def generate_figure(fs_id, fig_title, norms_path='./CSLB/norms.dat',
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
        Added to title after CSLB

    show_visual: bool
        Show swarmplot using matplotlib
    """

    # for feature fit
    categ = pd.read_csv(norms_path, sep='\t')
    categ = categ[['feature type', 'feature']]

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


def store_data(fs_id, fp_id, file_path):
    """
    fs_id: string
        Dict mapping between feature name and F1 score

    fp_id: string
        Numpy ndarray contain parameters for each feature model

    file_path: string
        Path where figures and progress files will be saved
    """

    score_df = pd.DataFrame(list(fs_id.items()), columns=['Feature', 'F1score'])

    # f, f1score, alpha, end_iter, intercept, rest_of_params
    params_data = {'Feature': fp_id[:, 0], 'Alpha': fp_id[:, 1], 'EndIter': fp_id[:, 2], 'Intercept': fp_id[:, 3],
                   'Coefs': list(fp_id[:, 4:])}

    params_df = pd.DataFrame(params_data, columns=['Feature', 'Alpha', 'EndIter', 'Intercept', 'Coefs'])

    df = score_df.merge(right=params_df, on='Feature')

    df = df.set_index('Feature')
    df.sort_index(inplace=True)
    logging.info('Saving: {}'.format(file_path))
    df.to_csv(file_path)


def figure_from_csv(path, fig_title, norms_path):
    df = pd.read_csv(path, index_col=0)
    d = dict(zip(df.index.tolist(), df.F1score.values))

    generate_figure(fs_id=d, fig_title=fig_title, norms_path=norms_path)


def evaluate_cslb(embedding, df_cleaned_cslb, n_jobs=4, nb_hyper=20, max_iter=1800, random_state=0):
    """
    Evaluate how well embedding encode features perceptual features. This experiment use CSLB semantic norms.

    Parameters
    ----------
    embedding: Embedding
        Loaded embedding

    df_cleaned_cslb: DataFrame
        Preprocessed pandas clsb dataframe. Columns names are concepts.

    n_jobs: int
        Numbers of threads used for learning

    nb_hyper: int
        Number of hyperparm value to select. Used by logistics regression for each feature independently

    max_iter: int
        Max iter of SGD

    random_state: int or RandomState
        Seed important for replicability

    Returns
    -------
    fs_id, fp_id: dict, ndarray
      Dict mapping between feature name and F1 score. Numpy ndarray contain parameters for each feature model.

    References
    ----------
        Reference paper: 'Are distributional representations ready for the real world? Evaluating word vectors for grounded perceptual meaning'
        https://arxiv.org/abs/1705.11168
    """

    logging.info('Shape of cleaned CSLB is {}'.format(df_cleaned_cslb.shape))
    cdf = df_cleaned_cslb.transpose()

    concepts = [str(x) for x in cdf.index]
    features = cdf.columns

    fs_id, fp_id = _learn_logit_reg(embedding=embedding, features=features, concepts=concepts,
                                    cleaned_norms=df_cleaned_cslb,
                                    n_jobs=n_jobs, random_state=random_state, max_iter=max_iter, nb_hyper=nb_hyper)

    return fs_id, fp_id
