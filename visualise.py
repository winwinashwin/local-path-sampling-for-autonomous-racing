"""Visualiser for debugging and testing."""

import json
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from path_sampling.core import (
    gen_spline, intersection_line_cubic, parametrise_lineseg
)
from path_sampling.global_path_handler import GlobalPathHandler
from path_sampling.types import Pose, RoadLinePolynom


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

closest_pt_idx = gp_handler.get_closest_point(ego_pose)
cls_pt_x, cls_pt_y, *_ = gp_handler.global_path.loc[closest_pt_idx]

ppd_line = gp_handler.get_perpendicular(closest_pt_idx, 20)
ppd_xs = np.linspace(-299, -305, 1000)

coeff_left = RoadLinePolynom(*tc_data['left'])
coeff_right = RoadLinePolynom(*tc_data['right'])

p1 = intersection_line_cubic(ppd_line, coeff_left)
p2 = intersection_line_cubic(ppd_line, coeff_right)

parametric_pt = parametrise_lineseg(p1, p2, padding=0.04)

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

# plt.plot([cls_pt_x], [cls_pt_y], marker='o', markersize=5)

# plt.plot(ppd_xs, [ppd_line.m * x + ppd_line.c for x in ppd_xs])

# plt.plot([p1.x], [p1.y], marker='o', markersize=5)
# plt.plot([p2.x], [p2.y], marker='o', markersize=5)

for e in np.linspace(0, 1, 10):
    p = parametric_pt(e)
    xs, ys = gen_spline(ego_pose, p)
    plt.plot(xs, ys, color='#4d79ff', linewidth=1, zorder=0)

plt.show()
