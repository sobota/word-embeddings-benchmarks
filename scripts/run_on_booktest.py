from multiprocessing.pool import Pool
import datetime
import logging
import optparse
import sys
import os
import re

from web.embedding import Embedding
from web.embeddings import *

from web.experiments.feature_view import evaluate_cslb, store_data, process_CSLB, generate_figure


def run(w, opts, title_part):
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

    f1, params = evaluate_cslb(w, df_cleaned_cslb=df_cleaned, n_jobs=opts.n_jobs, max_iter=opts.max_iter)

    logging.info('Generating Plots & Storing Data')

    now_dt = datetime.datetime.now()

    fig_path = os.path.join(opts.save_path,
                            'cslb_{}_{:%d-%m-%Y_%H:%M}.png'.format(re.sub('\s', '', title_part), now_dt))
    store_path = os.path.join(opts.save_path,
                              'cslb_f1_params_{}_{:%d-%m-%Y_%H:%M}.csv'.format(re.sub('\s', '', title_part),
                                                                               now_dt))

    store_data(f1, params, store_path)

    generate_figure(f1, fig_title=title_part, norms_path=cslb_norm, fig_path=fig_path)

    logging.info('End of experiment for {}'.format(title_part))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info('The feature view')

    parser = optparse.OptionParser()
    parser.add_option("-j", "--n_jobs", type="int", default=45)
    parser.add_option("-m", "--max_iter", type="int", default=2000)
    parser.add_option("-c", "--cslb_path", type="string", default='~')
    parser.add_option("-s", "--save_path", type="string", default='./')

    (opts, args) = parser.parse_args()

    # 1462682 dim = 100

    e = Embedding.from_word2vec('~/Storage/fastbooks-skipgram.vec')
    run(e, opts, 'BooksCorpora Skipgram')

    e = Embedding.from_word2vec('~/Storage/fastbooks-cbow.vec')
    run(e, opts, 'BooksCorpora Cbow')

    run(fetch_FastText(), opts, 'BooksCorpora fetch_FastText()')
