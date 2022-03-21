import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import napari
from skimage import measure, filters, data

# If you haven't installed tysserand with pip\nbut its code is on your hard drive:
import sys
sys.path.extend(['../../tysserand']) # change directory accordingly

from tysserand import tysserand as ty

# Generate some test data (smooth 3D blob shapes)
imgarray = filters.gaussian(np.squeeze(np.stack([data.binary_blobs(length=300, n_dim=3, blob_size_fraction=0.1, volume_fraction=0.01)[:, 0:256, 0:256]])).astype(float), sigma=(2.5, 2.5, 2.5))

# Open viewer (Qt window) with axes = slice, row, column
viewer = napari.Viewer(title='volume test', ndisplay=3)
# viewer.add_image(data=imgarray, name='blobs', scale=[256/300, 1, 1], rendering='attenuated_mip', attenuation=2.0, contrast_limits=(0.25, 1))
viewer.add_image(data=imgarray, name='blobs', rendering='attenuated_mip', attenuation=2.0);


masks = measure.label(imgarray>0.1, background=0)

coords = ty.mask_val_coord(masks)[['x', 'y', 'z']].values
pairs = ty.build_delaunay(coords)
# coords, pairs = ty.refactor_coords_pairs(coords, pairs)
distances = ty.distance_neighbors(coords, pairs)

nodes = ty.coords_to_df(coords)
edges = ty.pairs_to_df(pairs)

# enable multimodalities per node:
MULTI_MOD = False

# it's set at random, don't expect biological insight!
attributes = {'cell_type':['stromal', 'cancer', 'immune'],
              'marker':['PanCK', 'CD8', 'CD4', 'PDL-1', 'CTLA-4']}

nodes_att = pd.DataFrame(data=None, index=np.arange(coords.shape[0]))

if MULTI_MOD:
    for att_name, att_mod in attributes.items():
        att_val = np.random.randint(0, 2, size=(coords.shape[0],len(att_mod))).astype(bool)
        nodes_att = nodes_att.join(pd.DataFrame(att_val, columns=att_mod))
else:
    for att_name, att_mod in attributes.items():
        att_val = np.random.choice(att_mod, coords.shape[0])
        nodes_att = nodes_att.join(pd.DataFrame(att_val, columns=[att_name]))
nodes = nodes.join(nodes_att)
nodes.head()

# make colors for nodes
#                 orange      blue      green
class_colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
classes = list(nodes['cell_type'].unique())
dico_col = {classes[0]:class_colors[0],
            classes[1]:class_colors[1],
            classes[2]:class_colors[2]}
colors = []
for cl in nodes['cell_type']:
    colors.append(dico_col[cl])


# make colors for nodes
#                 orange      blue      green
class_colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
classes = list(nodes['cell_type'].unique())
dico_col = {classes[0]:class_colors[0],
            classes[1]:class_colors[1],
            classes[2]:class_colors[2]}

    
annotations = ty.make_annotation_dict(
    coords, pairs=pairs,
    nodes_class=nodes['cell_type'],
    nodes_class_color_mapper=dico_col,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tysserand import tysserand as ty
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

data_dir = Path("/home/mouneem/Projects/tysserand/tysserand/new_data/")
img = plt.imread(data_dir / "NVA_21-003.IMMCORE.C1v1_17T003826-06-ImvessC1-4518_job79795_MarkupMid.png")

fig, ax = ty.showim(img, figsize=(50, 50))


ty.add_annotations(viewer, annotations)
plt.show()