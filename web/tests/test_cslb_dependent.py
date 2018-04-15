import sys
import os
import logging
import numpy as np
import pandas as pd
from web.embeddings import fetch_GloVe, Embedding
from web.experiments.feature_view import process_CSLB, generate_figure, _learn_logit_reg


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

    clean = process_CSLB('./web/tests/data/mocked_feature_matrix.dat', emb)

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

    clean = process_CSLB('./web/tests/data/mocked_feature_matrix.dat', emb)

    cdf = clean.transpose()
    for f in cdf.columns:
        # is broken concept number
        assert np.sum(clean.loc[f].values) >= 5

    cslb_words = cdf.index
    logging.info(cslb_words)

    logging.info(set(emb.words).difference(cslb_words))

    for w in cslb_words:
        assert w in emb.words

    assert 'ring_(jewellery)' not in cslb_words
    assert 'ring' not in cslb_words


# def test_logreglearn():
#
#     words = ['accordion', 'aeroplane', 'alligator', 'ambulance', 'anchor', 'ant',
#              'apple', 'apricot', 'apron', 'arm', 'dragonfly', 'dress', 'dresser', 'dressing_gown', 'drill', 'drum',
#              'duck', 'eagle', 'ear', 'earmuffs', 'eel', 'elephant', 'elm', 'emu',
#              'encyclopaedia', 'envelope', 'eucalyptus', 'eye', 'falcon', 'fence', 'goggles', 'goldfish', 'gong',
#              'goose', 'gorilla', 'gown', 'grape', 'grapefruit', 'grasshopper', 'grater', 'greeting_card', 'grenade',
#              'guinea_pig', 'guitar', 'gun', 'hair', 'ham', 'hammer', 'hamster',
#              'harmonica', 'harp', 'harpoon', 'harpsichord', 'hatchet', 'hawk',
#              'heart', 'hedgehog', 'helicopter', 'helmet', 'heron', 'herring',
#              'hippo', 'hoe', 'hook', 'hornet', 'horse', 'hose', 'houseboat',
#              'housefly', 'hummingbird', 'hutch', 'hyacinth', 'hyena', 'ibuprofen',
#              'ice_cream', 'iguana', 'jacket', 'jam', 'jar', 'jeans', 'jeep', 'jelly',
#              'jellyfish', 'jug', 'kangaroo', 'kayak', 'ketchup', 'kettle', 'key',
#              'kingfisher', 'kitchen_scales', 'kite', 'kiwi_fruit', 'rake', 'range_rover', 'raspberry', 'rat', 'rattle',
#              'rattlesnake', 'raven', 'razor', 'recorder', 'revolver', 'rhino', 'rhubarb', 'rice',
#              'rifle', 'ring_(jewellery)', 'robe', 'robin', 'rock', 'rocket',
#              'rocking_chair', 'rollerskate', 'rolling_pin', 'rolls_royce', 'rope',
#              'rose', 'ruler', 'salmon', 'sandals', 'sandpaper', 'sandwich',
#              'sardine', 'satchel', 'satsuma', 'saw', 'saxophone', 'scallop',
#              'scalpel', 'scarf', 'scissors', 'scorpion', 'screw', 'screwdriver',
#              'scythe', 'seagull', 'seahorse', 'seashell', 'seaweed',
#              'sellotape', 'shark', 'shawl', 'sheep', 'shield', 'ship', 'shirt',
#              'shoes', 'shotgun', 'shovel', 'shrimp', 'sink', 'skateboard', 'skirt',
#              'skis', 'skunk', 'sledge', 'slippers', 'slug', 'snail', 'sock', 'sofa',
#              'soup', 'spade', 'spanner', 'sparrow', 'spatula', 'spear', 'speedboat',
#              'window', 'wine', 'wolf', 'woodpecker', 'worm', 'wren', 'yacht',
#              'yoghurt', 'yoyo', 'zebra']
#
#     features = ['comes_in_a_stick', 'does_DIY', 'does_absorb_water', 'does_add_air',
#                 'does_affect_urine', 'does_allow_movement', 'does_attach_things',
#                 'does_attract_attention', 'does_baa_bleat', 'does_bark',
#                 'does_float', 'does_flush', 'does_flutter', 'does_fly', 'does_fly_high',
#                 'does_fly_in_the_sky', 'does_fold', 'does_frolic', 'does_gallop',
#                 'does_give_choice', 'does_glide', 'does_gnaw', 'does_go_anywhere',
#                 'does_go_in_a_shoe', 'does_go_into_space', 'does_go_off',
#                 'does_go_on_roads', 'does_go_through_red_lights', 'does_go_to_hospital',
#                 'does_go_up_and_down', 'does_go_with_chairs', 'does_go_with_knives',
#                 'does_go_with_saucer', 'does_gobble', 'does_grate', 'does_grate_cheese',
#                 'does_grip', 'does_grow', 'does_grow_close_to_the_ground',
#                 'does_grow_from_a_bulb', 'does_grow_from_caterpillars',
#                 'does_grow_in_ground', 'does_grow_in_hot_countries',
#                 'does_grow_on_bushes', 'does_grow_on_plants', 'does_grow_on_trees',
#                 'does_grow_on_vines', 'does_grow_underground', 'does_growl',
#                 'does_guide_larger_boats', 'does_hang', 'does_hang_from_ceilings',
#                 'does_hang_from_something', 'does_hang_upside_down', 'does_heat',
#                 'does_heat_water', 'does_help_balance', 'does_help_people_speak',
#                 'does_hibernate', 'does_hiss', 'does_hit', 'does_hit_balls',
#                 'does_hit_nails', 'does_hit_people', 'does_hold_ash', 'does_hold_books',
#                 'does_hold_contain_liquid_water', 'does_hold_drinks', 'does_hold_food',
#                 'made_of_vinegar', 'made_of_water_is_watery', 'made_of_wax',
#                 'made_of_wheat', 'made_of_wicker', 'made_of_wire', 'made_of_wood',
#                 'made_of_wood_and_metal', 'made_of_wool', 'made_of_yeast', 'does_throw_rocks', 'does_tick', 'does_tidy',
#                 'does_tie', 'does_tie_around_back', 'does_tighten', 'does_tighten_bolts',
#                 'does_toast', 'does_toast_bread', 'does_tow', 'does_translate',
#                 'does_transport_cars', 'does_transport_oil', 'does_trap', 'does_travel',
#                 'does_travel_long_distances', 'does_treat_illness',
#                 'does_treat_infections', 'does_trot', 'does_trumpet', 'does_turn',
#                 'does_twang', 'does_twist', 'does_type', 'does_untangle_hair',
#                 'does_use_echolocation', 'does_use_electricity',
#                 'does_use_fuel_diesel_petrol', 'does_use_gas', 'does_use_water',
#                 'does_vibrate', 'does_waddle', 'does_wag_its_tail', 'does_wake_you_up',
#                 'does_walk', 'does_walk_sideways', 'does_warm', 'does_wash_clothes',
#                 'does_washing_up', 'does_weigh', 'does_whip', 'does_whip_cream',
#                 'does_whisk', 'does_whistle', 'does_work', 'does_wrap_around',
#                 'does_wriggle', 'found_in_bars', 'has_a_baby_seat', 'has_a_back',
#                 'has_a_bag', 'has_a_bag_of_air', 'has_a_balance', 'has_a_ball',
#                 'has_a_barrel', 'has_a_base', 'has_a_basin']
#
#     vecs = np.random.normal(loc=1.0, scale=1.0, size=(len(words), 100))
#
#     # fake embedding
#     w_vec = dict(zip(words, vecs))
#
#     emb = Embedding.from_dict(w_vec)
#
#     n, p = 1, .34  # number of trials, probability of each trial
#
#     rows = {w: np.random.binomial(n, p, size=len(features)) for w in words}
#
#     features = list(map(lambda f: f.replace('_', ' '), features))
#     cleaned = pd.DataFrame(data=rows, index=features)
#
#     f_f1score, _ = _learn_logit_reg(embedding=emb, cleaned_norms=cleaned, features=features, concepts=words, n_jobs=45,
#                                     max_iter=700,
#                                     nb_hyper=2)
#
#     save_fig_path = './test_logreg.png'
#     if os.path.isfile(save_fig_path):
#         os.remove(save_fig_path)
#     # todo
#     _generate_figure(f_f1score, 'TEST FIGURE', norms_path='./data/mocked_norms.dat', fig_path=save_fig_path)
#
#     # probably better way for validation chart?
#     assert os.path.isfile(save_fig_path)
#     assert np.max(np.asarray(list(f_f1score.values())).astype(np.float32)) <= 1.0
#     assert np.min(np.asarray(list(f_f1score.values())).astype(np.float32)) >= 0


def test_figure_generation():
    fetaures = ['comes_in_a_stick', 'does_DIY', 'does_absorb_water', 'does_add_air',
                'does_affect_urine', 'does_allow_movement', 'does_attach_things',
                'does_attract_attention', 'does_baa_bleat', 'does_bark',
                'does_float', 'does_flush', 'does_flutter', 'does_fly', 'does_fly_high',
                'does_fly_in_the_sky', 'does_fold', 'does_frolic', 'does_gallop',
                'does_give_choice', 'does_glide', 'does_gnaw', 'does_go_anywhere',
                'does_go_in_a_shoe', 'does_go_into_space', 'does_go_off',
                'does_go_on_roads', 'does_go_through_red_lights', 'does_go_to_hospital',
                'does_go_up_and_down', 'does_go_with_chairs', 'does_go_with_knives',
                'does_go_with_saucer', 'does_gobble', 'does_grate', 'does_grate_cheese',
                'does_grip', 'does_grow', 'does_grow_close_to_the_ground',
                'does_grow_from_a_bulb', 'does_grow_from_caterpillars',
                'does_grow_in_ground', 'does_grow_in_hot_countries',
                'does_grow_on_bushes', 'does_grow_on_plants', 'does_grow_on_trees',
                'does_grow_on_vines', 'does_grow_underground', 'does_growl',
                'does_guide_larger_boats', 'does_hang', 'does_hang_from_ceilings',
                'does_hang_from_something', 'does_hang_upside_down', 'does_heat',
                'does_heat_water', 'does_help_balance', 'does_help_people_speak',
                'does_hibernate', 'does_hiss', 'does_hit', 'does_hit_balls',
                'does_hit_nails', 'does_hit_people', 'does_hold_ash', 'does_hold_books',
                'does_hold_contain_liquid_water', 'does_hold_drinks', 'does_hold_food',
                'made_of_vinegar', 'made_of_water_is_watery', 'made_of_wax',
                'made_of_wheat', 'made_of_wicker', 'made_of_wire', 'made_of_wood',
                'made_of_wood_and_metal', 'made_of_wool', 'made_of_yeast', 'does_throw_rocks', 'does_tick', 'does_tidy',
                'does_tie', 'does_tie_around_back', 'does_tighten', 'does_tighten_bolts',
                'does_toast', 'does_toast_bread', 'does_tow', 'does_translate',
                'does_transport_cars', 'does_transport_oil', 'does_trap', 'does_travel',
                'does_travel_long_distances', 'does_treat_illness',
                'does_treat_infections', 'does_trot', 'does_trumpet', 'does_turn',
                'does_twang', 'does_twist', 'does_type', 'does_untangle_hair',
                'does_use_echolocation', 'does_use_electricity',
                'does_use_fuel_diesel_petrol', 'does_use_gas', 'does_use_water',
                'does_vibrate', 'does_waddle', 'does_wag_its_tail', 'does_wake_you_up',
                'does_walk', 'does_walk_sideways', 'does_warm', 'does_wash_clothes',
                'does_washing_up', 'does_weigh', 'does_whip', 'does_whip_cream',
                'does_whisk', 'does_whistle', 'does_work', 'does_wrap_around',
                'does_wriggle', 'found_in_bars', 'has_a_baby_seat', 'has_a_back',
                'has_a_bag', 'has_a_bag_of_air', 'has_a_balance', 'has_a_ball',
                'has_a_barrel', 'has_a_base', 'has_a_basin']

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    f1 = np.random.uniform(0, 1, size=len(fetaures))

    feature_f1 = dict(zip(fetaures, f1))

    save_fig_path = './test_fig.png'

    if os.path.isfile(save_fig_path):
        os.remove(save_fig_path)

    generate_figure(feature_f1, norms_path='./web/tests/data/mocked_norms.dat', fig_path=save_fig_path,
                    show_visual=False,
                    fig_title='TEST_FIGURE_GENERATION')

    # todo better way for validation chart?
    assert os.path.isfile(save_fig_path)
