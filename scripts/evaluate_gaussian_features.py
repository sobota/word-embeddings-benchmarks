from multiprocessing.pool import Pool
import datetime
import logging
import optparse
import sys

from web.experiments.gaussian_embedding import cslb_on_gaussian_experiment

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info('The feature view on Gaussian')

    parser = optparse.OptionParser()
    parser.add_option("-j", "--n_jobs", type="int", default=45)
    parser.add_option("-m", "--max_iter", type="int", default=1800)
    parser.add_option("-c", "--cslb_path", type="string", default='../CSLB')
    parser.add_option("-s", "--save_path", type="string", default='./')
    (opts, args) = parser.parse_args()

    logging.info('Started at {:%d-%m-%Y %H:%M}'.format(datetime.datetime.now()))

    # Gaussian noise for baseline
    dims = [25, 50, 100, 300]

    n_jobs = opts.n_jobs


    def run_job(d):
        title_part = 'Gaussian Noise dim={}'.format(d)
        cslb_on_gaussian_experiment(d, figure_desc=title_part, cslb_path=opts.cslb_path, save_path=opts.save_path,
                                    n_jobs=n_jobs, max_iter=opts.max_iter)

        logging.info('End of experiment for {}'.format(title_part))


    Pool(n_jobs).map(run_job, dims)
