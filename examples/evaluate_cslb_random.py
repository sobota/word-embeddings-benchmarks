from multiprocessing.pool import Pool
import datetime
import logging
import optparse
import sys
import os
import re

from web import embeddings

from web.experiments.feature_view import *


def _learn_logit_reg_50_to_50(embedding, features, concepts, cleaned_norms, n_jobs=4, random_state=None, nb_hyper=20,
                              max_iter=None):
    X = np.asarray([embedding[c_w] for c_w in concepts])

    # features in semantic norms

    def train(f):
        y = cleaned_norms.loc[f].values

        # np.count_nonzero(y == 1)

        y_rand = np.random.randint(0, 2, y.size)

        l = y_rand.size
        s = sum(y_rand)
        f1s = f1_score(y_true=y, y_pred=y_rand)

        # f, f1score, alpha, end_iter, intercept, rest_of_params
        return f, f1s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    data = list(map(train, tqdm(features, desc='Train logreg for each feature')))

    data = np.asarray(data)

    features_f1_scored = dict(zip(data[:, 0], data[:, 1]))
    features_params = np.delete(data, np.s_[1], axis=1)

    return features_f1_scored, features_params


def _learn_logit_reg_balanced(embedding, features, concepts, cleaned_norms, n_jobs=4, random_state=None, nb_hyper=20,
                              max_iter=None):
    X = np.asarray([embedding[c_w] for c_w in concepts])

    def generate_balnced_y(y):
        pos = np.count_nonzero(y == 1)
        neg = np.count_nonzero(y == 0)

        assert pos + neg == y.size

        y_val = np.zeros(0, y.size)

        pos_count = 0
        while pos_count < pos:

            idx = np.random.randint(0, y.size)

            if y_val[idx] == 0:
                y_val[idx] = 1
                pos_count += 1

        return y_val

    # features in semantic norms
    def train(f):
        y = cleaned_norms.loc[f].values

        y_rand = generate_balnced_y(y)

        f1s = f1_score(y_true=y, y_pred=y_rand)

        # f, f1score, alpha, end_iter, intercept, rest_of_params
        return f, f1s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    data = list(map(train, tqdm(features, desc='Train logreg for each feature')))

    data = np.asarray(data)

    features_f1_scored = dict(zip(data[:, 0], data[:, 1]))
    features_params = np.delete(data, np.s_[1], axis=1)

    return features_f1_scored, features_params


# learn

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info('The feature view')

    parser = optparse.OptionParser()
    parser.add_option("-j", "--n_jobs", type="int", default=45)
    parser.add_option("-m", "--max_iter", type="int", default=2000)
    parser.add_option("-c", "--cslb_path", type="string", default='../CSLB')
    parser.add_option("-s", "--save_path", type="string", default='./')

    (opts, args) = parser.parse_args()

    logging.info('Started at {:%d-%m-%Y %H:%M}'.format(datetime.datetime.now()))

    jobs = []

    # GloVe
    for dim in [50]:  # , 100, 200, 300]:
        jobs.append(["fetch_GloVe", {"dim": dim, "corpus": "wiki-6B"}])

    for dim in [25, 50, 100, 200]:
        jobs.append(["fetch_GloVe", {"dim": dim, "corpus": "twitter-27B"}])

    for corpus in ["common-crawl-42B", "common-crawl-840B"]:
        jobs.append(["fetch_GloVe", {"dim": 300, "corpus": corpus}])

    # PDC and HDC
    for dim in [50, 100, 300]:
        jobs.append(["fetch_PDC", {"dim": dim}])
        jobs.append(["fetch_HDC", {"dim": dim}])

    # SG
    jobs.append(["fetch_SG_GoogleNews", {}])

    # LexVec
    jobs.append(["fetch_LexVec", {}])

    # ConceptNet Numberbatch
    jobs.append(["fetch_conceptnet_numberbatch", {}])

    # FastText
    jobs.append(["fetch_FastText", {}])

    # Word2Bits
    jobs.append(['fetch_Word2Bits', {}])

    n_jobs = opts.n_jobs


    def run_job(j):
        fn, kwargs = j

        w = getattr(embeddings, fn)(**kwargs)
        title_part = '{}({})'.format(fn, kwargs)

        cslb_matrix = os.path.join(opts.cslb_path, 'feature_matrix.dat')
        cslb_norm = os.path.join(opts.cslb_path, 'norms.dat')

        if not os.path.isfile(cslb_matrix):
            print("Download CSLB file first from website: http://www.csl.psychol.cam.ac.uk/propertynorms/",
                  file=sys.stderr)
            raise FileNotFoundError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), cslb_matrix)

        if not os.path.isfile(cslb_norm):
            print("Download CSLB file first from website: http://www.csl.psychol.cam.ac.uk/propertynorms/",
                  file=sys.stderr)
            raise FileNotFoundError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), cslb_norm)

        df_cleaned = process_CSLB(cslb_matrix, w)

        logging.info('Shape of cleaned CSLB is {}'.format(df_cleaned.shape))
        cdf = df_cleaned.transpose()

        concepts = [str(x) for x in cdf.index]
        features = cdf.columns

        # f1, parms = _learn_logit_reg_50_to_50(embedding=w, features=features, concepts=concepts,
        #                                       cleaned_norms=df_cleaned,
        #                                       n_jobs=n_jobs, random_state=0, max_iter=1800, nb_hyper=20)

        f1, parms = _learn_logit_reg_balanced(embedding=w, features=features, concepts=concepts,
                                              cleaned_norms=df_cleaned,
                                              n_jobs=n_jobs, random_state=0, max_iter=1800, nb_hyper=20)

        logging.info('Generating Plots & Storing Data')

        now_dt = datetime.datetime.now()

        fig_path = os.path.join(opts.save_path,
                                'cslb_random_balanced_{}_{:%d-%m-%Y_%H:%M}.png'.format(re.sub('\s', '', title_part),
                                                                                     now_dt))
        store_path = os.path.join(opts.save_path,
                                  'cslb_random_balanced_f1_params_{}_{:%d-%m-%Y_%H:%M}.csv'.format(
                                      re.sub('\s', '', title_part),
                                      now_dt))

        store_data(f1, parms, store_path)

        generate_figure(f1, fig_title=title_part, norms_path=cslb_norm, fig_path=fig_path)

        logging.info('End of experiment for {}'.format(title_part))


    Pool(n_jobs).map(run_job, jobs)
