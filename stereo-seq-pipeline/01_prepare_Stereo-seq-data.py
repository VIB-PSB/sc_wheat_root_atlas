# ##############################################################################
# Prepare STEREO-seq data
# ##############################################################################

# Images are not well aligned with the GEM. Here, images are cropped around each
# section using optimal coordinates and the GEM is filtered accordingly.

# ==============================================================================
# Setup
# ==============================================================================

## Load libraries (kernel: napari-sparrow)

import os
import numpy as np
import pandas as pd
from skimage import io

from napari_sparrow import functions as fc

import win32com.client
shell = win32com.client.Dispatch("WScript.Shell")

## -----------------------------------------------------------------------------
## Define input/output
## -----------------------------------------------------------------------------

### Root paths (from workspace level)
paths = {"root": "."}
paths.update({"data_raw": shell.CreateShortCut(paths["root"] + "/data_raw.lnk").Targetpath,
              "data_pro": paths["root"] + "/data_processed",
              "data_inp": paths["root"] + "/data_input"})
paths.update({"exp1"    : paths["data_raw"] + "/wheat-root_STEREO-seq_exp-1_raw",
              "exp2"    : paths["data_raw"] + "/wheat-root_STEREO-seq_exp-2_raw",
              "output"  : paths["data_pro"] + "/008_data-preparation"})

## Input
inputs = {
    # Data frame with coordinates for cropping/filtering
    "coordinates":   paths["data_inp"] + "/sections-coordinates.csv",
    
    # Experiment 1
    # --------------------------------------------------------------------------
    # Original image of cell-wall staining
    "exp1_image-wall":    paths["exp1"] + "/SS200000745BL_F6_FB_20230109_regist.tif",
    # Gene expression matrix
    "exp1_gem":           paths["exp1"] + "/SS200000745BL_F6.raw.gem",
    
    # Experiment 2
    # --------------------------------------------------------------------------
    # Original image of cell-wall staining
    "exp2_image-wall":    paths["exp2"] + "/C01828B1_regist/C01828B1_FB_regist_v2.tif",
    # Original image of nucleus staining (ssDNA)
    "exp2_image-nucleus": paths["exp2"] + "/C01828B1_regist/C01828B1_regist_v0.tif",
    # Gene expression matrix
    "exp2_gem":           paths["exp2"] + "/C01828B1.raw.gem",
    }

# ==============================================================================
# Execute
# ==============================================================================

## Create output directory
os.makedirs(paths["output"], exist_ok=True)

## -----------------------------------------------------------------------------
## Load data
## -----------------------------------------------------------------------------

### Raw images as numpy ndarray (cell-wall and nuclei)
exp1_img_wall_raw_np    = io.imread(inputs["exp1_image-wall"])
exp2_img_wall_raw_np    = io.imread(inputs["exp2_image-wall"])
exp2_img_nucleus_raw_np = io.imread(inputs["exp2_image-nucleus"])

### GEM as pandas dataframe
exp1_gem_df = fc.read_in_stereoSeq(inputs["exp1_gem"], skiprows = 6, xcol = 'x', ycol = 'y').compute()
exp2_gem_df = fc.read_in_stereoSeq(inputs["exp2_gem"], skiprows = 6, xcol = 'x', ycol = 'y').compute()

### Coordinates
coordinates = pd.read_csv(inputs["coordinates"])

## -----------------------------------------------------------------------------
## Select sections
## -----------------------------------------------------------------------------

### Function: crop image
def crop_image(image, coordinates):
    return image[coordinates["row_or_y_start"]:coordinates["row_or_y_end"],
                 coordinates["col_or_x_start"]:coordinates["col_or_x_end"]]

### Function: filter GEM
def filter_gem(gem, coordinates):

    # Subset GEM
    sub = gem[
              (gem["x"] >= coordinates["col_or_x_start"]) &
              (gem["x"] <= coordinates["col_or_x_end"  ]) &
              (gem["y"] >= coordinates["row_or_y_start"]) &
              (gem["y"] <= coordinates["row_or_y_end"  ])
             ].copy().reset_index(drop = True)
    
    ## Reset coordinates
    sub.loc[:, "x"] -= coordinates["col_or_x_start"]
    sub.loc[:, "y"] -= coordinates["row_or_y_start"]
    
    return sub

### For each section
for section_id in coordinates["Section_ID"].unique():
    # Prepare data
    coor       = coordinates[coordinates["Section_ID"] == section_id]
    experiment = section_id.split("_")[0]
    coor_gem   = coor[coor["Type"] == "gem"]
    coor_wall  = coor[coor["Type"] == "wall"]
    image_wall = globals()[f"{experiment}_img_wall_raw_np"]
    gem        = globals()[f"{experiment}_gem_df"]
    if experiment == "exp2":
        coor_nucleus  = coor[coor["Type"] == "nucleus"]
        image_nucleus = globals()[f"{experiment}_img_nucleus_raw_np"]
    
    # Crop cell-wall image
    img_crop_wall = crop_image(image_wall, coor_wall.iloc[0])
    
    # Crop nucleus image
    if experiment == "exp2":
        img_crop_nucleus = crop_image(image_nucleus, coor_nucleus.iloc[0])
    
    # Subset GEM
    gem_sub = filter_gem(gem, coor_gem.iloc[0])
    
    # Write files
    io.imsave(paths["output"] + f"/{section_id}_wall.tif", img_crop_wall)
    if experiment == "exp2":
        io.imsave(paths["output"] + f"/{section_id}_nucleus.tif", img_crop_nucleus)
    gem_sub.to_csv(paths["output"] + f"/{section_id}_gem.csv.gz", index = False)

    # # Visualize
    # plt.imshow(img_crop_wall, cmap = "gray")
    # plt.imshow(img_crop_nucleus, cmap = "magma", alpha = 0.5)
    # # df = gem_sub[gem_sub["gene"] == "TraesCS2B03G1501600"]  # TraesCS3B03G1120300
    # # plt.scatter(data = df, x = "x", y = "y", c = "gold", alpha = 0.1)
    # plt.show()
