import pandas as pd
import logging
import os
import sys

from web.experiments.feature_view import store_data


def test_filegen():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    d1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    d2 = {'a': 10, 'b': 20, 'c': 30, 'd': 40}

    save_file_path = './test_data.csv'
    if os.path.isfile(save_file_path):
        os.remove(save_file_path)

    store_data(d1, d2, file_path=save_file_path)

    df = pd.read_csv(save_file_path, index_col=0)

    assert df is not None
    assert df.shape == (4, 2)
