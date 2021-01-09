"""Visualiser for debugging and testing."""

import json
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from path_sampling.global_path_handler import GlobalPathHandler
from path_sampling.types import Pose, RoadLinePolynom
from path_sampling.spline import LateralSpG


logging.basicConfig(level=logging.INFO)

ego_pose = Pose(-300, -1890, yaw=1.57)

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

spline_gen = LateralSpG(gp_handler)

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

for xs, ys in spline_gen.generate(10, ego_pose, coeff_left, coeff_right, 20, 0.04):
    plt.plot(xs, ys, color='#4d79ff', linewidth=1, zorder=0)

plt.show()
