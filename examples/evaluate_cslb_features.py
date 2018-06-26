from multiprocessing.pool import Pool
import datetime
import logging
import optparse
import sys
import os
import re

from web import embeddings

from web.experiments.feature_view import evaluate_cslb, store_data, process_CSLB, generate_figure

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
    # for dim in [50, 100, 200, 300]:
    #     jobs.append(["fetch_GloVe", {"dim": dim, "corpus": "wiki-6B"}])

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
    # jobs.append(["fetch_FastText", {}])
    #
    # Word2Bits
    # jobs.append(['fetch_Word2Bits', {}])

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

        f1, params = evaluate_cslb(w, df_cleaned_cslb=df_cleaned, n_jobs=n_jobs, max_iter=opts.max_iter)

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


    Pool(n_jobs).map(run_job, jobs)
