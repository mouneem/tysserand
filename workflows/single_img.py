import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tysserand import tysserand as ty
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

# where we save and load annotations
data_dir = Path("/home/mouneem/Projects/tysserand/tysserand/new_data/")

nodes = pd.read_csv(data_dir / 'minNVA_21-003.IMMCORE.C1v1_17T003826-06-ImvessC1-4518.czi_79795_ObjData.fcs.csv', usecols=['x','y', 'class'])
print(len(nodes))
nodes = nodes.sample(n = round(len(nodes) / 2) )
print(len(nodes))

## Adjust
img = plt.imread(data_dir / "NVA_21-003.IMMCORE.C1v1_17T003826-06-ImvessC1-4518_job79795_MarkupMid.png")

from PIL import Image, ImageOps
#img = ImageOps.grayscale(img)

coords = nodes.loc[:,['x','y']].values

class_colors = ['#F85446', '#813FE0', '#52DEF7', '#62E03F', '#FFCE36', '#FA6C17',
               '#D805E3', '#0155FA', '#0BE35B', '#FFF117', '#DDFF7D', '#78E3C8']
classes = list(nodes['class'].unique())
print(classes)
dico_col = {classes[0]:class_colors[0],
            classes[1]:class_colors[1],
            classes[2]:class_colors[2],
            classes[3]:class_colors[3],
            classes[4]:class_colors[4],
            classes[5]:class_colors[5],
            classes[6]:class_colors[6],
            classes[7]:class_colors[7],
            classes[8]:class_colors[8],
            classes[9]:class_colors[9],
       #     classes[10]:class_colors[10],
          #  classes[11]:class_colors[11],
           }
colors = []
for cl in nodes['class']:
    colors.append(dico_col[cl])
    

    
    
min_x = min(coords[:, 0])
min_y = min(coords[:, 1])

max_x = max(coords[:, 0])
max_y = max(coords[:, 1])


print(min_x, min_y)

import cv2
#img = cv2.flip(img, 0)

import numpy

coords2 = coords

coords_x_diff = max(coords2[:, 0]) - min(coords2[:, 0])
coords_y_diff = max(coords2[:, 1]) - min(coords2[:, 1])

img2 = img
#img2 = cv2.resize(img, (coords_x_diff , coords_y_diff)) 
w = img.shape[0]
h = img.shape[1]

ratio_x = h / coords_x_diff
ratio_y = w / coords_y_diff

ratio = 1
ratio = 0.1
ratio = ratio_y


coords2[:, 0] = coords2[:, 0] * ratio
coords2[:, 1] = coords2[:, 1] * ratio

coords2[:, 0] = coords2[:, 0] - min(coords2[:, 0])
coords2[:, 1] = coords2[:, 1] - min(coords2[:, 1])

print(coords_x_diff , coords_y_diff)

pairs = ty.build_delaunay(coords2)
print(len(pairs))

coords = coords2
from libpysal.cg.voronoi  import voronoi, voronoi_frames

regions, vertices = voronoi(coords)
region_df, point_df = voronoi_frames(coords)

imdir = Path(".")
fig, ax = plt.subplots(figsize=(200,200))
region_df.plot(ax=ax, color='blue',edgecolor='black', alpha=0.3)
point_df.plot(ax=ax, color='red')
plt.axis('off')
plt.savefig(imdir.joinpath('mIF - PySAL Voronoi tessellation'), bbox_inches=('tight'))

ty.showim(color.label2rgb(masks, bg_label=0, colors=label_cmap), origin='lower')
plt.savefig(imdir.joinpath('generated-tissue-masks'), bbox_inches=('tight'))

    
#########################################################################
# import napari
# viewer = napari.Viewer()
# ty.visualize(viewer, img, colormaps='rgb')

# annotations = ty.make_annotation_dict(
#     coords, pairs=pairs,
#     nodes_class=nodes['class'],
#     nodes_class_color_mapper=dico_col,
# )
# ty.add_annotations(viewer, annotations)
# plt.show()