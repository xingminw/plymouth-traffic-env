import cv2
import numpy as np
from glob import glob
from traffic_envs.config import output_particle_folder


fpx = 2
whole_road_files = glob(output_particle_folder + "/zoom*")
file_nums = [int(val.split("zoom")[-1][:-4]) for val in whole_road_files]

file_nums = np.sort(file_nums).tolist()
file_list = [output_particle_folder + "/zoom" + str(val) + ".png" for val in file_nums]

# fetch size
height, width, _ = cv2.imread(file_list[0]).shape
print(height, width)

frame_list = []
out = cv2.VideoWriter("figs/zoom.avi", cv2.VideoWriter_fourcc(*"DIVX"), fpx, (width, height))

for file_name in file_list:
    img = cv2.imread(file_name)
    out.write(img)
out.release()

whole_road_files = glob(output_particle_folder + "/full*")
file_nums = [int(val.split("full")[-1][:-4]) for val in whole_road_files]

file_nums = np.sort(file_nums).tolist()
file_list = [output_particle_folder + "/full" + str(val) + ".png" for val in file_nums]

# fetch size
height, width, _ = cv2.imread(file_list[0]).shape
print(height, width)

frame_list = []
out = cv2.VideoWriter("figs/full.avi", cv2.VideoWriter_fourcc(*"DIVX"), fpx, (width, height))

for file_name in file_list:
    img = cv2.imread(file_name)
    out.write(img)
out.release()
