"""Visualiser for debugging and testing."""
import json
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


def main():
    plt.figure(figsize=(10, 10))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-285, -315)
    plt.ylim(-1810, -2200)

    plt.scatter(df_tl['LeftBnd_X'], df_tl['LeftBnd_Y'], s=4, color='#505050')
    plt.scatter(df_tl['RightBnd_X'], df_tl['RightBnd_Y'], s=4, color='#505050')
    plt.scatter(df_gp['X'], df_gp['Y'], s=2, color='#cc0000')

    plt.gca().add_patch(Rectangle(
        (ego_pose.x + 0.25, ego_pose.y + 5),
        -0.5, -10,
        facecolor='#000'
    ))
    plt.gca().add_patch(Rectangle(
        (obs_pose.x + 0.25, obs_pose.y + 5),
        -0.5, -10,
        facecolor='#000'
    ))

    for xs, ys in spline_gen.generate_lat(100, padding=0.05, bias=0.5, pts_per_spline=100):
        plt.plot(xs, ys, color='#4d79ff', linewidth=1, zorder=0)

    for xs, ys in spline_gen.generate_long(25, density=1, bias=0.8, pts_per_spline=100):
        plt.plot(xs, ys, color='#12961A', linewidth=1, zorder=0)

    plt.show()


if __name__ == '__main__':
    main()
