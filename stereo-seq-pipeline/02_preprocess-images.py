# ##############################################################################
# Pre-process image
# ##############################################################################

# The cell-wall image is preprocessed using napari-sparrow in order to increase
# the quality for segmentation. A new image is exported.

# ==============================================================================
# Setup
# ==============================================================================

## Load libraries (kernel: napari-sparrow)

import os
import numpy as np
import pandas as pd

from skimage import io
import squidpy as sq
from napari_sparrow import functions as fc

## -----------------------------------------------------------------------------
## Define input/output
## -----------------------------------------------------------------------------

### Root paths
paths = {"root": "."}  # from workspace level
paths.update({"data_pro": paths["root"] + "/data_processed",
              "data_inp": paths["root"] + "/data_input"})

paths.update({"images": paths["data_pro"] + "/008_data-preparation"})
paths.update({"output": paths["data_pro"] + "/010_sparrow-pipeline/preprocessed-images"})

## General inputs
general = {"sections": paths["data_inp"] + "/sections.csv"}

## Section-specific inputs
sections              = pd.read_csv(general["sections"])
sections["Image_raw"] = paths["images"] + "/" + sections["Section_ID"] + "_wall.tif"

## Section-specific outputs
sections["Image_preprocess"] = paths["output"] + "/" + sections["Section_ID"] + "_wall_preprocess.tif"

## Parameters
parameters = {"size_tophat"  : 200,
              "contrast_clip": 1}

# ==============================================================================
# Execute
# ==============================================================================

## Create output directory
os.makedirs(paths["output"], exist_ok=True)

## -----------------------------------------------------------------------------
## Load data
## -----------------------------------------------------------------------------

images_raw = {}
for index, row in sections.iterrows():
    section_ID = row["Section_ID"]
    file       = row["Image_raw"]
    images_raw[section_ID] = sq.im.ImageContainer(io.imread(file))

## -----------------------------------------------------------------------------
## Pre-process image
## -----------------------------------------------------------------------------

images_preprocess = {}
# For each section image
for section_ID, image in images_raw.items():
    # Pre-process
    image_preprocess = fc.preprocessImage(
        img           = image,
        size_tophat   = parameters["size_tophat"],
        contrast_clip = parameters["contrast_clip"]
        )
    # Output
    images_preprocess[section_ID] = image_preprocess
    # Save
    image_preprocess_np = image_preprocess.data.image.squeeze().to_numpy()
    file = sections[sections["Section_ID"] == section_ID]["Image_preprocess"].iloc[0]
    io.imsave(file, image_preprocess_np)
