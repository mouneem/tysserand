#Load libs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 
from PIL import Image, ImageOps
import fcsparser
import sys

#set up working dir
directory = "/home/abdelmounimessabbar/Bureau/PanCK/ObjectData"

outpath = "/home/abdelmounimessabbar/Bureau/PanCK/"
log = "/home/abdelmounimessabbar/Bureau/PanCK/"+'log.txt'

# FCS TO CSV
# iterate over files in
# that directory
i = 0
failed = []
for filename in os.listdir(directory):
    i += 1
    f = os.path.join(directory, filename)
    # checking if it is a file
    filename_csv = filename
    if os.path.isfile(f) :
        #meta, data = fcsparser.parse(f, reformat_meta=True)
        data = pd.read_csv(f) 
        #df1 = df[['a', 'b']] // df1 = df.iloc[:, 0:2] 
        image_name = data.iat[0,0].split("\\")[-1].split("/")[-1]
        level = data.iat[0,1]
        outname = ".".join([ image_name,level,'csv' ])
        data["x"] = (data["XMax"] + data["XMin"] ) / 2
        data["y"] = (data["YMax"] + data["YMin"] ) / 2

        col_is_marker = []
        for col in list(data.columns):
            if col.count('+') + col.count('-') > 1:
                col_is_marker.append(True)
            else:
                col_is_marker.append(False)
