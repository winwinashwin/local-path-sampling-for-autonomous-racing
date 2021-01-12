"""Benchmark spline generation."""

import time
import json
import logging
import pandas as pd

from path_sampling.global_path_handler import GlobalPathHandler
from path_sampling.types import Pose, RoadLinePolynom
from path_sampling.spline import SplineGenerator

logging.basicConfig(level=logging.INFO)

ego_pose = Pose(-300, -1890, yaw=1.57)
obs_pose = Pose(-302.54, -1950.3, yaw=0)

tl_file = 'testing/data/track_limits.csv'
gp_file = 'testing/data/global_path.csv'
tc_file = 'testing/data/track_limits_coeffs.json'

df_tl = pd.read_csv(tl_file)
df_gp = pd.read_csv(gp_file)
tc_data = json.loads(open(tc_file).read())

gp_handler = GlobalPathHandler()
gp_handler.load_from_csv(gp_file)

coeff_left = RoadLinePolynom(*tc_data['left'])
coeff_right = RoadLinePolynom(*tc_data['right'])

spline_gen = SplineGenerator(gp_handler, ego_pose, obs_pose, coeff_left, coeff_right)


def timer_func(func):
    def function_timer(*args, **kwargs):
        n_trials = 100
        runtime = 0.0
        for i in range(n_trials):
            print('Iteration --', i + 1, end='\r')
            start = time.time()
            value = func(*args, **kwargs)
            end = time.time()
            runtime += end - start

        msg = 'Function: {func} took {time} seconds'
        print(msg.format(func=func.__name__, time=runtime / n_trials))
        return value

    return function_timer


@timer_func
def spline_generation():
    for xs, ys in spline_gen.generate_lat(100, padding=0.05, bias=0.5, pts_per_spline=100):
        pass

    for xs, ys in spline_gen.generate_long(20, density=1, bias=0.8, pts_per_spline=100):
        pass


if __name__ == '__main__':
    spline_generation()
