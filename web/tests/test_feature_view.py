import numpy as np
import logging
import os
import sys

from web.experiments.feature_view import generate_figure


def test_figure_generation():
    f = ['comes_in_a_stick', 'does_DIY', 'does_absorb_water', 'does_add_air',
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
    f1 = np.random.uniform(0, 1, size=len(f))

    feature_f1 = dict(zip(f, f1))

    save_fig_path = './test_fig.png'

    if os.path.isfile(save_fig_path):
        os.remove(save_fig_path)

    generate_figure(feature_f1, norms_path='../experiments/CSLB/norms.dat', fig_path=save_fig_path, show_visual=False,
                    fig_title='TEST_FIGURE_GENERATION')

    # todo better way for validation chart?
    assert os.path.isfile(save_fig_path)
