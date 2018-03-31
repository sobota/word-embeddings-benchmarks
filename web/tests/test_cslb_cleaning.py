import numpy as np
import logging
import sys

from web.embeddings import fetch_GloVe, Embedding
from web.experiments.feature_view import process_CSLB


def test_cleaning_cslb():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    words = ['accordion', 'aeroplane', 'alligator', 'ambulance', 'anchor', 'ant',
             'apple', 'apricot', 'apron', 'arm', 'dragonfly', 'dress', 'dresser', 'dressing_gown', 'drill', 'drum',
             'duck', 'eagle', 'ear', 'earmuffs', 'eel', 'elephant', 'elm', 'emu',
             'encyclopaedia', 'envelope', 'eucalyptus', 'eye', 'falcon', 'fence', 'goggles', 'goldfish', 'gong',
             'goose', 'gorilla', 'gown', 'grape', 'grapefruit', 'grasshopper', 'grater', 'greeting_card', 'grenade',
             'guinea_pig', 'guitar', 'gun', 'hair', 'ham', 'hammer', 'hamster',
             'harmonica', 'harp', 'harpoon', 'harpsichord', 'hatchet', 'hawk',
             'heart', 'hedgehog', 'helicopter', 'helmet', 'heron', 'herring',
             'hippo', 'hoe', 'hook', 'hornet', 'horse', 'hose', 'houseboat',
             'housefly', 'hummingbird', 'hutch', 'hyacinth', 'hyena', 'ibuprofen',
             'ice_cream', 'iguana', 'jacket', 'jam', 'jar', 'jeans', 'jeep', 'jelly',
             'jellyfish', 'jug', 'kangaroo', 'kayak', 'ketchup', 'kettle', 'key',
             'kingfisher', 'kitchen_scales', 'kite', 'kiwi_fruit', 'rake', 'range_rover', 'raspberry', 'rat', 'rattle',
             'rattlesnake', 'raven', 'razor', 'recorder', 'revolver', 'rhino', 'rhubarb', 'rice',
             'rifle', 'ring_(jewellery)', 'robe', 'robin', 'rock', 'rocket',
             'rocking_chair', 'rollerskate', 'rolling_pin', 'rolls_royce', 'rope',
             'rose', 'ruler', 'salmon', 'sandals', 'sandpaper', 'sandwich',
             'sardine', 'satchel', 'satsuma', 'saw', 'saxophone', 'scallop',
             'scalpel', 'scarf', 'scissors', 'scorpion', 'screw', 'screwdriver',
             'scythe', 'seagull', 'seahorse', 'seashell', 'seaweed',
             'sellotape', 'shark', 'shawl', 'sheep', 'shield', 'ship', 'shirt',
             'shoes', 'shotgun', 'shovel', 'shrimp', 'sink', 'skateboard', 'skirt',
             'skis', 'skunk', 'sledge', 'slippers', 'slug', 'snail', 'sock', 'sofa',
             'soup', 'spade', 'spanner', 'sparrow', 'spatula', 'spear', 'speedboat',
             'window', 'wine', 'wolf', 'woodpecker', 'worm', 'wren', 'yacht',
             'yoghurt', 'yoyo', 'zebra']

    vecs = np.random.normal(loc=1.0, scale=1.0, size=(len(words), 100))

    # fake embedding
    w_vec = dict(zip(words, vecs))

    emb = Embedding.from_dict(w_vec)

    clean = process_CSLB(emb, feature_matrix_path='../experiments/CSLB/feature_matrix.dat')

    cdf = clean.transpose()
    for f in cdf.columns:
        # is broken concept number
        assert sum(clean.loc[f].values) >= 5

    cslb_words = cdf.index
    logging.info(cslb_words)

    for w in cslb_words:
        assert w in words

    assert 'ring_(jewellery)' not in cslb_words
    assert 'ring' not in cslb_words


def test_on_real_embedding():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    emb = fetch_GloVe(corpus='wiki-6B', dim=50)

    clean = process_CSLB(emb, feature_matrix_path='../experiments/CSLB/feature_matrix.dat')

    cdf = clean.transpose()
    for f in cdf.columns:
        # is broken concept number
        assert sum(clean.loc[f].values) >= 5

    cslb_words = cdf.index
    logging.info(cslb_words)

    logging.info(set(emb.words).difference(cslb_words))

    for w in cslb_words:
        assert w in emb.words

    assert 'ring_(jewellery)' not in cslb_words
    assert 'ring' not in cslb_words
