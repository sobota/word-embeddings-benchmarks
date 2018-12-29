from __future__ import print_function
import datetime
import logging
import optparse
import os
import re
import sys

from tqdm import tqdm

from web import embeddings
from web import perceptual as pc

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    logging.info('Perceptual evaluation')

    parser = optparse.OptionParser()
    parser.add_option("-j", "--n_jobs", type="int", default=45)
    parser.add_option("-c", "--cslb_path", type="string", default='../CSLB')
    parser.add_option("-s", "--save_path", type="string", default='./')

    (opts, args) = parser.parse_args()

    logging.info('Started at {:%d-%m-%Y %H:%M}'.format(datetime.datetime.now()))

    jobs = []

    # GloVe
    for dim in [50, 100, 200, 300]:
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
    #
    # FastText
    jobs.append(["fetch_FastText", {}])
    #
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

        df_cleaned = pc.process_CSLB(cslb_matrix, w)

        f1, params = pc.evaluate(w, df_cleaned_cslb=df_cleaned, n_jobs=n_jobs)

        logging.info('Generating Plots & Storing Data')

        now_dt = datetime.datetime.now()

        fig_path = os.path.join(opts.save_path,
                                'cslb_{}_{:%d-%m-%Y_%H:%M}.png'.format(re.sub('\s', '', title_part), now_dt))
        store_path = os.path.join(opts.save_path,
                                  'cslb_f1_params_{}_{:%d-%m-%Y_%H:%M}.csv'.format(re.sub('\s', '', title_part),
                                                                                   now_dt))

        pc.store_data(f1, params, store_path)

        pc.generate_figure(f1, fig_title=title_part, norms_path=cslb_norm, fig_path=fig_path)

        logging.info('End of experiment for {}'.format(title_part))


    for j in tqdm(jobs):
        run_job(j)
