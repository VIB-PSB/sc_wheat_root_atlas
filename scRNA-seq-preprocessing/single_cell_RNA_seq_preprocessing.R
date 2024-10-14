library(DoubletFinder)
library(tidyverse)
library(Seurat)
library(patchwork)

# Load the wheat_root_1 dataset
CSwheat_root_1 <- Read10X(data.dir = "~/path_to_dataset/filtered_feature_bc_matrix")
# Initialize the Seurat object with the raw (non-normalized data).
CSwheat_root_1 <- CreateSeuratObject(counts = CSwheat_root_1, project = "wheat-root-1", min.cells = 3, min.features = 200)
CSwheat_root_1
# Filter out low-quality cells
CSwheat_root_1[["percent.mt"]] <- PercentageFeatureSet(CSwheat_root_1, pattern = "mitogenome")
# Define chloroplast reads
CSwheat_root_1[["percent.chloro"]] <- PercentageFeatureSet(CSwheat_root_1, pattern = "chlorogenome")
# Visualize QC metrics as a violin plot
VlnPlot(CSwheat_root_1, features = c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.chloro"), ncol = 4)
# FeatureScatter is typically used to visualize feature-feature relationships, but can be used
# for anything calculated by the object, i.e. columns in object metadata, PC scores etc.
plot1 <- FeatureScatter(CSwheat_root_1, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(CSwheat_root_1, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2
# Filter out low-quality cells
CSwheat_root_1 <- subset(CSwheat_root_1, subset = nFeature_RNA > 500 & nCount_RNA > 800 & percent.mt < 10 & percent.chloro < 5)
str(CSwheat_root_1)
CSwheat_root_1 <- SCTransform(CSwheat_root_1)
CSwheat_root_1 <- RunPCA(CSwheat_root_1, verbose = F)
ElbowPlot(CSwheat_root_1)
pc.num=1:30
CSwheat_root_1 <- RunUMAP(CSwheat_root_1, dims=pc.num)
CSwheat_root_1 <- FindNeighbors(CSwheat_root_1, dims = pc.num) %>% FindClusters(resolution = 0.3)
DimPlot(CSwheat_root_1)
sweep.res.list <- paramSweep_v3(CSwheat_root_1, PCs = pc.num, sct = T)
sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
pK_bcmvn <- bcmvn$pK[which.max(bcmvn$BCmetric)] %>% as.character() %>% as.numeric()
DoubletRate = 0.014                     
homotypic.prop <- modelHomotypic(CSwheat_root_1$seurat_clusters)  
nExp_poi <- round(DoubletRate*ncol(CSwheat_root_1))
nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
CSwheat_root_1 <- doubletFinder_v3(CSwheat_root_1, PCs = pc.num, pN = 0.25, pK = pK_bcmvn,
                                   nExp = nExp_poi.adj, reuse.pANN = F, sct = T)
colnames(CSwheat_root_1@meta.data)[ncol(CSwheat_root_1@meta.data)]="DoubletFinder"
table(CSwheat_root_1@meta.data$DoubletFinder)
DimPlot(CSwheat_root_1,reduction = "umap",pt.size = 1,group.by = "DoubletFinder")
CSwheat_root_1_filtered_singlets <- subset(CSwheat_root_1, cells=rownames(CSwheat_root_1@meta.data)[which(CSwheat_root_1@meta.data$DoubletFinder == "Singlet")])
DimPlot(CSwheat_root_1_filtered_singlets, reduction = "umap", pt.size = 1)

# Load the wheat_root_2 dataset
CSwheat_root_2 <- Read10X(data.dir = "/path_to_dataset/filtered_feature_bc_matrix")
# Initialize the Seurat object with the raw (non-normalized data).
CSwheat_root_2 <- CreateSeuratObject(counts = CSwheat_root_2, project = "wheat-root-2", min.cells = 3, min.features = 200)
CSwheat_root_2
# Filter out low-quality cells
CSwheat_root_2[["percent.mt"]] <- PercentageFeatureSet(CSwheat_root_2, pattern = "mitogenome")
# Define chloroplast reads
CSwheat_root_2[["percent.chloro"]] <- PercentageFeatureSet(CSwheat_root_2, pattern = "chlorogenome")
# Visualize QC metrics as a violin plot
VlnPlot(CSwheat_root_2, features = c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.chloro"), ncol = 4)
# FeatureScatter is typically used to visualize feature-feature relationships, but can be used
# for anything calculated by the object, i.e. columns in object metadata, PC scores etc.
plot1 <- FeatureScatter(CSwheat_root_2, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(CSwheat_root_2, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2
# Filter out low-quality cells
CSwheat_root_2 <- subset(CSwheat_root_2, subset = nFeature_RNA > 500 & nCount_RNA > 800 & percent.mt < 10 & percent.chloro < 5)
str(CSwheat_root_2)
CSwheat_root_2 <- SCTransform(CSwheat_root_2)
CSwheat_root_2 <- RunPCA(CSwheat_root_2, verbose = F)
ElbowPlot(CSwheat_root_2)
pc.num=1:30
CSwheat_root_2 <- RunUMAP(CSwheat_root_2, dims=pc.num)
CSwheat_root_2 <- FindNeighbors(CSwheat_root_2, dims = pc.num) %>% FindClusters(resolution = 0.3)
DimPlot(CSwheat_root_2)
sweep.res.list <- paramSweep_v3(CSwheat_root_2, PCs = pc.num, sct = T)
sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
pK_bcmvn <- bcmvn$pK[which.max(bcmvn$BCmetric)] %>% as.character() %>% as.numeric()
DoubletRate = 0.026                    
homotypic.prop <- modelHomotypic(CSwheat_root_2$seurat_clusters)   
nExp_poi <- round(DoubletRate*ncol(CSwheat_root_2))
nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
CSwheat_root_2 <- doubletFinder_v3(CSwheat_root_2, PCs = pc.num, pN = 0.25, pK = pK_bcmvn,
                                   nExp = nExp_poi.adj, reuse.pANN = F, sct = T)
colnames(CSwheat_root_2@meta.data)[ncol(CSwheat_root_2@meta.data)]="DoubletFinder"
table(CSwheat_root_2@meta.data$DoubletFinder)
DimPlot(CSwheat_root_2,reduction = "umap",pt.size = 1,group.by = "DoubletFinder")
CSwheat_root_2_filtered_singlets <- subset(CSwheat_root_2, cells=rownames(CSwheat_root_2@meta.data)[which(CSwheat_root_2@meta.data$DoubletFinder == "Singlet")])
DimPlot(CSwheat_root_2_filtered_singlets, reduction = "umap", pt.size = 1)

# Load the wheat_root_3 dataset
CSwheat_root_3 <- Read10X(data.dir = "~/path_to_dataset/filtered_feature_bc_matrix")
# Initialize the Seurat object with the raw (non-normalized data).
CSwheat_root_3 <- CreateSeuratObject(counts = CSwheat_root_3, project = "wheat-root-3", min.cells = 3, min.features = 200)
CSwheat_root_3
# Filter out low-quality cells
CSwheat_root_3[["percent.mt"]] <- PercentageFeatureSet(CSwheat_root_3, pattern = "mitogenome")
# Define chloroplast reads
CSwheat_root_3[["percent.chloro"]] <- PercentageFeatureSet(CSwheat_root_3, pattern = "chlorogenome")
# Visualize QC metrics as a violin plot
VlnPlot(CSwheat_root_3, features = c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.chloro"), ncol = 4)
# FeatureScatter is typically used to visualize feature-feature relationships, but can be used
# for anything calculated by the object, i.e. columns in object metadata, PC scores etc.
plot1 <- FeatureScatter(CSwheat_root_3, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(CSwheat_root_3, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2
# Filter out low-quality cells
CSwheat_root_3 <- subset(CSwheat_root_3, subset = nFeature_RNA > 500 & nCount_RNA > 800 & percent.mt < 10 & percent.chloro < 5)
str(CSwheat_root_3)
CSwheat_root_3 <- SCTransform(CSwheat_root_3)
CSwheat_root_3 <- RunPCA(CSwheat_root_3, verbose = F)
ElbowPlot(CSwheat_root_3)
pc.num=1:30
CSwheat_root_3 <- RunUMAP(CSwheat_root_3, dims=pc.num)
CSwheat_root_3 <- FindNeighbors(CSwheat_root_3, dims = pc.num) %>% FindClusters(resolution = 0.3)
DimPlot(CSwheat_root_3)
sweep.res.list <- paramSweep_v3(CSwheat_root_3, PCs = pc.num, sct = T)
sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
pK_bcmvn <- bcmvn$pK[which.max(bcmvn$BCmetric)] %>% as.character() %>% as.numeric()
DoubletRate = 0.022                     
homotypic.prop <- modelHomotypic(CSwheat_root_3$seurat_clusters)   
nExp_poi <- round(DoubletRate*ncol(CSwheat_root_3))
nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
CSwheat_root_3 <- doubletFinder_v3(CSwheat_root_3, PCs = pc.num, pN = 0.25, pK = pK_bcmvn,
                                   nExp = nExp_poi.adj, reuse.pANN = F, sct = T)
colnames(CSwheat_root_3@meta.data)[ncol(CSwheat_root_3@meta.data)]="DoubletFinder"
table(CSwheat_root_3@meta.data$DoubletFinder)
DimPlot(CSwheat_root_3,reduction = "umap",pt.size = 1,group.by = "DoubletFinder")
CSwheat_root_3_filtered_singlets <- subset(CSwheat_root_3, cells=rownames(CSwheat_root_3@meta.data)[which(CSwheat_root_3@meta.data$DoubletFinder == "Singlet")])
DimPlot(CSwheat_root_3_filtered_singlets, reduction = "umap", pt.size = 1)

# Check if there's any batch effect by merging.

root_1<-readRDS("~/path_to_dataset/CSwheat_root_1_filtered_singlets.rds")
root_2<-readRDS("~/path_to_dataset/CSwheat_root_2_filtered_singlets.rds")
root_3<-readRDS("~/path_to_dataset/CSwheat_root_3_filtered_singlets.rds")


object.combined<-merge(x=root_1,y=c(root_2,root_3),add.cell.ids=c("root_1","root_2","root_3"),project="wheat-root")

object.combined<-SCTransform(object.combined)


object.combined <- RunPCA(object = object.combined, features = VariableFeatures(object.combined),
                          
                          npcs = 150, assay = "SCT",
                          
                          reduction.name = "SCT_pca",reduction.key = "sctPC_", verbose = F)


p1<-DimPlot(object.combined,reduction="SCT_pca", group.by = "orig.ident")

p2<-VlnPlot(object.combined,features="sctPC_1",group.by="orig.ident",pt.size = 0)

plot(p1)
plot(p2)

library(harmony)
library(Seurat)
library(tidyverse)
library(ggplot2)
CS_root.combined<-merge(x=CSwheat_root_1_filtered_singlets,y=c(CSwheat_root_2_filtered_singlets,CSwheat_root_3_filtered_singlets),add.cell.ids=c("CS1","CS2","CS3"),project="CS")
CS_root.combined <- SCTransform(CS_root.combined)
CS_root.combined <- RunPCA(CS_root.combined)
ElbowPlot(CS_root.combined)
CS_root.combined <- RunUMAP(CS_root.combined, dims = 1:30, reduction = 'pca')
before <- DimPlot(CS_root.combined, reduction = 'umap', group.by = 'orig.ident')
before

CS_root.harmony <- CS_root.combined %>%
  RunHarmony(group.by.vars = 'orig.ident', plot_convergence = FALSE, assay.use = "SCT")
CS_root.harmony@reductions
CS_root.harmony.embed <- Embeddings(CS_root.harmony, "harmony")
CS_root.harmony.embed[1:10,1:10]
CS_root.harmony <- CS_root.harmony %>%
  RunUMAP(reduction = 'harmony', dims = 1:30) %>%
  FindNeighbors(reduction = "harmony", dims = 1:30) %>%
  FindClusters(resolution = 0.24)
after <- DimPlot(CS_root.harmony, reduction = 'umap', group.by = 'orig.ident')
before|after
saveRDS(CS_root.harmony, file = "~/path_to_dataset/CS_root.harmony_res0.24.rds")
CS_harmony<-readRDS("~/path_to_dataset/CS_root.harmony_res0.24.rds")

