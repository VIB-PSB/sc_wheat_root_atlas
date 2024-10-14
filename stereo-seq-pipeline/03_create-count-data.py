# ##############################################################################
# Create count data
# ##############################################################################

# For each section, takes a GEM, a pre-processed image and manually-defined
# cells and combines them in an AnnData object. Does some filtering and returns:
# - The AnnData with image and count data.
# - The polygons for manually-defined cells, with initial cell-types.
# - The initial cell-types.
# - The count matrix and corresponding genes and cells names.

# Note: sparrow is used that, but it is not optimal to use dask dataframe on .csv.gz files.

# ==============================================================================
# Setup
# ==============================================================================

## Load libraries (kernel: napari-sparrow)

import os
import numpy as np
import pandas as pd

import dask.dataframe as dd
from skimage import io
import squidpy as sq
import re
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio

from napari_sparrow import functions as fc

from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import gzip

## -----------------------------------------------------------------------------
## Define input/output
## -----------------------------------------------------------------------------

### Root paths
paths      = {"root": "."}  # from workspace level
paths.update({"data_pro": paths["root"] + "/data_processed",
              "data_inp": paths["root"] + "/data_input"})

paths.update({"gem"   : paths["data_pro"] + "/008_data-preparation",
              "images": paths["data_pro"] + "/010_sparrow-pipeline/preprocessed-images",
              "cells" : paths["data_inp"] + "/manually-defined-cells",
              "output": paths["data_pro"] + "/010_sparrow-pipeline/count-data"})

### General inputs
general = {"sections": paths["data_inp"] + "/sections.csv"}

### Section-specific inputs
sections             = pd.read_csv(general["sections"])
sections["GEM"]      = paths["gem"]          + "/" + sections["Section_ID"] + "_gem.csv.gz"
sections["Image"]    = paths["images"]       + "/" + sections["Section_ID"] + "_wall_preprocess.tif"

# List of files with manually-defined cells per cell-type
cells = []
for section_ID in sections["Section_ID"]:
    files = [paths["cells"] + "/" + filename for filename in os.listdir(paths["cells"]) if filename.startswith(section_ID)]
    cells.append(files)
sections["Cells"] = cells
del section_ID, files, cells

### Section-specific outputs
sections["Anndata"]      = paths["output"] + "/" + sections["Section_ID"] + "_anndata.h5ad"
sections["Polygons"]     = paths["output"] + "/" + sections["Section_ID"] + "_polygons.geojson"
sections["Cell-types"]   = paths["output"] + "/" + sections["Section_ID"] + "_initial-celltypes.csv"
sections["Count_matrix"] = paths["output"] + "/" + sections["Section_ID"] + "_count-data_matrix.mtx"
sections["Count_cells"]  = paths["output"] + "/" + sections["Section_ID"] + "_count-data_cells.csv"
sections["Count_genes"]  = paths["output"] + "/" + sections["Section_ID"] + "_count-data_genes.csv"

# ==============================================================================
# Execute
# ==============================================================================

## Create output directory
os.makedirs(paths["output"], exist_ok=True)

## -----------------------------------------------------------------------------
## Load data
## -----------------------------------------------------------------------------

gem_by_section               = {}
image_by_section             = {}
cells_by_celltype_by_section = {}
# For each section
for index, row in sections.iterrows():
    # Get section ID
    section_ID                   = row["Section_ID"]
    # Load GEM (use dask for compatibility with sparrow)
    gem_by_section[section_ID]   = dd.read_csv(row["GEM"], compression = "gzip", blocksize=None)
    # Load image
    image_by_section[section_ID] = sq.im.ImageContainer(io.imread(row["Image"]))
    # Load manually-defined cells for each cell-type
    cells_by_celltype = {}
    for celltype_file in row["Cells"]:
        celltype = re.sub(r".*_|\.csv$", "", celltype_file)
        cells_by_celltype[celltype] = pd.read_csv(celltype_file)
    cells_by_celltype_by_section[section_ID] = cells_by_celltype
del index, row, section_ID, celltype_file, celltype

## -----------------------------------------------------------------------------
## Filter transcripts to protein-coding
## -----------------------------------------------------------------------------

# For each section gem
for section_ID, gem in gem_by_section.items():
    gem_by_section[section_ID] = gem[~gem["gene"].str.contains("LC")]
del section_ID, gem

## -----------------------------------------------------------------------------
## Convert manually-defined cells to polygons and mask
## -----------------------------------------------------------------------------

### Convert to polygons
cells_polygons_by_section = {}
# For each section
for section_ID, cells_by_celltype in cells_by_celltype_by_section.items():
    cells_polygons = gpd.GeoDataFrame()
    # For each cell-type
    for celltype, cells in cells_by_celltype.items():
        # Convert cells to polygons
        cells    = cells.groupby("index")
        polygons = []
        for group_name, group_data in cells:
            points  = list(zip(group_data["axis-1"], group_data["axis-0"]))
            polygon = Polygon(points)
            polygons.append(polygon)
        polygons = gpd.GeoDataFrame(polygons, columns = ["geometry"])
        # Add manually-defined cell-type as column
        polygons["initial-celltype"] = celltype
        # Append
        cells_polygons = pd.concat([cells_polygons, polygons], ignore_index=True)
    # Add cell ID
    cells_polygons["cell_ID"] = cells_polygons.index + 1
    # Append
    cells_polygons_by_section[section_ID] = cells_polygons
del section_ID, cells_by_celltype
del celltype, cells, polygons, group_name, group_data, points, polygon, cells_polygons

### Convert to mask
cells_mask_by_section = {}
# For each section
for section_ID, cells_polygons in cells_polygons_by_section.items():
    # Get image dimensions
    image         = image_by_section[section_ID].data.image.squeeze().to_numpy()
    width, height = image.shape
    # Create mask from polygons
    shapes = zip(cells_polygons["geometry"], cells_polygons["cell_ID"])
    mask   = rasterio.features.rasterize(shapes, out_shape = (width, height))
    # Append
    cells_mask_by_section[section_ID] = mask
del section_ID, cells_polygons, image, width, height, shapes, mask

## -----------------------------------------------------------------------------
## Create AnnData with transcripts allocated to cells
## -----------------------------------------------------------------------------

# Note about Anndata outputs
# adata.obs["cell_ID"]   - Cells ID as pandas Series
# adata.obsm["spatial"]  - Cells coordinates, initially as numpy ndarray (later as pandas DataFrame)
# adata.obsm["polygons"] - Cells polygons, initially as geopandas GeoDataFrame (changes later)
# adata.uns["spatial"]   - Notably image in ...[library_id]["images"]["hires"]

# Note about polygons
# The function below recreates polygons from the mask. Cell IDs remain the same.

adata_by_section               = {}
gene_count_per_cell_by_section = {}
# For each section
for section_ID in sections["Section_ID"]:
    # Create AnnData
    adata, df = fc.allocation(ddf        = gem_by_section[section_ID],
                              ic         = image_by_section[section_ID],
                              masks      = cells_mask_by_section[section_ID],
                              library_id = "spatial_transcriptomics",
                              radius     = 0,
                              polygons   = None
                              )
    # Output
    adata_by_section[section_ID]               = adata
    gene_count_per_cell_by_section[section_ID] = df
del section_ID, adata, df

### Add initial cell-types to polygons
# For each section
for section_ID, adata in adata_by_section.items():
    # Get adata polygons
    polygons_adata = adata.obsm["polygons"].copy()
    # Get original polygons and prepare for joining
    polygons_original       = cells_polygons_by_section[section_ID].copy()
    polygons_original.index = polygons_original["cell_ID"]
    polygons_original.index = polygons_original.index.astype('str')
    # Join
    polygons = polygons_adata.join(polygons_original[["initial-celltype"]], how = "left")
    # Update adata polygons with initial cell-types
    adata_by_section[section_ID].obsm["polygons"] = polygons
del section_ID, adata, polygons_adata, polygons_original, polygons

## -----------------------------------------------------------------------------
## Save
## -----------------------------------------------------------------------------

# For each section
for index, row in sections.iterrows():
    
    # Prepare data
    section_ID = row["Section_ID"]
    adata      = adata_by_section[section_ID].copy()
    
    # Save polygons as geojson
    adata.obsm["polygons"].to_file(row["Polygons"], driver = "GeoJSON")
    
    # Save initial cell-types
    celltypes = adata.obsm["polygons"]["initial-celltype"].copy()
    celltypes.index.name = "cell_ID"
    celltypes.to_csv(row["Cell-types"], header = True, index = True)
    
    # Save AnnData as h5ad (remove polygons as no method for geometry)
    del adata.obsm["polygons"]
    adata.write(row["Anndata"])
    
    # Save count matrix as compressed sparse Matrix Market
    mtx        = adata.X.astype(int)
    mtx_sparse = csr_matrix(mtx)
    mmwrite(row["Count_matrix"], mtx_sparse)
    with open(row["Count_matrix"], "rb") as f_in, gzip.open(row["Count_matrix"] + ".gz", "wb") as f_out:
        f_out.writelines(f_in)
    os.remove(row["Count_matrix"])
    
    # Save corresponding cells names
    adata.obs["cell_ID"].to_csv(row["Count_cells"], header = False, index = False)
    
    # Save corresponding gene names
    adata.var.to_csv(row["Count_genes"], header = False)

del index, row, section_ID, adata, mtx, mtx_sparse, f_in, f_out
