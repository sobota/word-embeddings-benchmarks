from multiprocessing.pool import Pool
import datetime
import logging
import optparse
import sys

from web import embeddings

from web.experiments.feature_view import cslb_experiment

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info('The feature view')

    parser = optparse.OptionParser()
    parser.add_option("-j", "--n_jobs", type="int", default=45)
    parser.add_option("-m", "--max_iter", type="int", default=1800)
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

    # FastText
    jobs.append(["fetch_FastText", {}])

    # Word2Bits
    jobs.append(['fetch_Word2Bits', {}])

    n_jobs = opts.n_jobs


    def run_job(j):
        fn, kwargs = j

        w = getattr(embeddings, fn)(**kwargs)
        title_part = '{}({})'.format(fn, kwargs)
        cslb_experiment(w, figure_desc=title_part, cslb_path=opts.cslb_path, save_path=opts.save_path, n_jobs=n_jobs,
                        max_iter=opts.max_iter)

        logging.info('End of experiment for {}'.format(title_part))


    Pool(n_jobs).map(run_job, jobs)
