#!/usr/bin/env python3

#############################################
### CROSS-SPECIES OVERLAP ANALYSIS MODULE ###
#############################################

from collections import defaultdict

import pandas as pd
import numpy as np


def read_files(file_locs, sep=",", include_orthogroups=True):
    
    """
    Read DEG (Differentially Expressed Genes) files for multiple species and create a dictionary.

    Parameters:
        file_locs (dict): A dictionary with keys as species and values as the location of DEG files for each species.
        sep (str): The delimiter used in the DEG files. Default is ",".
        include_orthogroups (bool): Whether to include orthogroups in the output dataframes. Default is True.

    Returns:
        dict: A dictionary with keys as species and values as dataframes containing selected columns:
              - If include_orthogroups is True: ["cluster", "gene ID", "Orth_group", "avg_log2FC", "p_val_adj"]
              - If include_orthogroups is False: ["cluster", "gene ID", "avg_log2FC", "p_val_adj"]
              Each dataframe contains the corresponding DEG information from the input files.
    """
    
    species2df_dict = dict()
    
    # For each given species, read in the DEG file
    for species, file_loc in file_locs.items():
        deg_df = pd.read_csv(file_loc, sep=sep, dtype=str)
        
        # Remove NAs (due to no orthogroup, or no gene conversion)
        if include_orthogroups:
            deg_df = deg_df.dropna(subset=["Orth_group", "gene ID"])
        else:
            deg_df = deg_df.dropna(subset=["gene ID"])
        
        # Select the cluster, gene and orthogroup columns and add to dict
        if include_orthogroups:
            species2df_dict[species] = deg_df[["cluster", "gene ID", "Orth_group", "avg_log2FC", "p_val_adj"]]
        else:
            species2df_dict[species] = deg_df[["cluster", "gene ID", "avg_log2FC", "p_val_adj"]]
    
    return species2df_dict


def convert_df_to_dict(species2df_dict, include_orthogroups=True, remove_duplicate_genes=False):
    
    """
    Convert DEG (Differentially Expressed Genes) dataframes to a structured dictionary format.

    Parameters:
        species2df_dict (dict): A dictionary with keys as species and values as dataframes containing
                                DEG information with columns "cluster", "gene ID", and optionally "Orth_group".
        include_orthogroups (bool): A flag indicating whether to include orthogroups in the output dictionaries. Default is True.
        remove_duplicate_genes (bool): A flag indicating whether to remove duplicate gene IDs within the same cluster.
                                       Default is False.

    Returns:
        tuple: A tuple containing two elements:
            - species2gene2group (dict or None): A dictionary with keys as species and values as dictionaries mapping gene IDs to Orthogroups. 
                                                 Returns None if include_orthogroups is False.
            - species2cluster2genes_and_groups (dict): A dictionary with keys as species and values as dictionaries
                                                       mapping cluster names to sets of genes and optionally orthogroups.
    """
    
    species2cluster2genes_and_groups = dict()
    if include_orthogroups:
        species2gene2group = dict()
    
    # For each given species, convert the DEG dataframe
    for species, df in species2df_dict.items():
        
        # Put all the gene-to-group relations in a dict
        if include_orthogroups:
            species2gene2group[species] = dict(zip(df["gene ID"], df["Orth_group"]))
        
        # Structure the data per cluster in a dict
        cluster2genes_and_groups = dict()

        # Add the genes and groups of the individual clusters to the dict
        for cluster in set(df.cluster):

            # Select only the rows of a given cluster
            if include_orthogroups:
                cluster_df = df.query("cluster == @cluster")[["cluster", "gene ID", "Orth_group"]]
            else:
                cluster_df = df.query("cluster == @cluster")[["cluster", "gene ID"]]
            
            # Remove duplicate genes in the same cluster (might be introduced when changing gene IDs to another version)
            cluster_df = cluster_df.drop_duplicates()
            
            # If explicitly stated, allow removal of duplicates
            if remove_duplicate_genes:
                cluster_df = cluster_df.drop_duplicates("gene ID")

            # Extract the genes and groups as sets
            genes = set(cluster_df["gene ID"])
            assert len(genes) == len(cluster_df["gene ID"]) # check for duplicated DEGs in one cluster
            if include_orthogroups:
                groups = set(cluster_df["Orth_group"])

            # Add them to the dict
            if include_orthogroups:
                cluster2genes_and_groups[cluster] = {"genes": genes, "groups": groups}
            else:
                cluster2genes_and_groups[cluster] = {"genes": genes}

        # Add the genes and groups of all clusters together to the dict
        if include_orthogroups:
            cluster2genes_and_groups["all_clusters"] = {"genes": set(df["gene ID"]), "groups": set(df["Orth_group"])}
        else:
            cluster2genes_and_groups["all_clusters"] = {"genes": set(df["gene ID"])}
        
        # Add to species dict
        species2cluster2genes_and_groups[species] = cluster2genes_and_groups
    
    if include_orthogroups:
        return species2gene2group, species2cluster2genes_and_groups
    else:
        return None, species2cluster2genes_and_groups


def load_gene_conversion(conversion_file):
    """
    Load maize gene conversion information, allowing for one-to-many relationships.

    Parameters:
        conversion_file (str): Path to the gene conversion file, which contains columns "new" and "old"
                               for the new and old gene IDs, respectively.

    Returns:
        tuple: A tuple containing two dictionaries:
            - gene_old2new (dict): A dictionary mapping old gene IDs to lists of new gene IDs.
            - gene_new2old (dict): A dictionary mapping new gene IDs to lists of old gene IDs.
    """
    
    # Load the gene conversion information
    conversion = pd.read_csv(conversion_file, sep="\t", names=["new", "old"])
    conversion = conversion.query("new != 'None' and old != 'None'")
    
    # Initialize dictionaries to store the mappings
    gene_old2new, gene_new2old = defaultdict(list), defaultdict(list)
    
    # Add old-to-new and new-to-old mappings to the dictionaries
    for old_gene, new_gene in zip(conversion["old"], conversion["new"]):
        gene_old2new[old_gene].append(new_gene)
        gene_new2old[new_gene].append(old_gene)
    
    # Convert defaultdict back to regular dict
    gene_old2new, gene_new2old = dict(gene_old2new), dict(gene_new2old)
    
    return gene_old2new, gene_new2old


def read_miniex_files_convert_IDs(file_locs, gene2group_dict, species_to_id_conversion, sep=","):
    
    """
    Read MINI-EX ranked regulons files for multiple species, convert gene IDs, and create a dictionary of dataframes.

    Parameters:
        file_locs (dict): A dictionary where keys are species names and values are file paths to the MINI-EX ranked 
                          regulons files for each species.
        gene2group_dict (dict): A dictionary where keys are gene IDs and values are the orthogroups they belong to.
        species_to_id_conversion (dict): A dictionary where keys are species names and values are dictionaries that map 
                                         old gene IDs to lists of new gene IDs.
        sep (str): The delimiter used in the ranked regulons files. Default is ",".

    Returns:
        dict: A dictionary where keys are species names and values are dataframes containing selected columns 
              ('cluster', 'gene ID', 'Orth_group') from the corresponding MINI-EX ranked regulons files.
    """
    
    species2df_dict = dict()
    
    # For each given species, read in the DEG file
    for species, file_loc in file_locs.items():
        deg_df = pd.read_csv(file_loc, sep=sep, dtype=str)
        
        # Convert the Borda rank from string to numeric
        deg_df["borda_rank"] = pd.to_numeric(deg_df["borda_rank"])
        
        # Remove the added cluster prefixes, to get the original cluster name
        deg_df["cluster"] = deg_df.cluster.str.split("_Cluster_").str[-1]
        
        # Rename the TF column to gene ID to be consistent with the earlier defined format
        deg_df = deg_df.rename(columns={"TF":"gene ID"})
        
        # Convert old genome version IDs to the new ones (if multiple in the list due to split genes, simply take the first one)
        if species in species_to_id_conversion:
            new_genes = [species_to_id_conversion[species][old_gene][0] \
                         if old_gene in species_to_id_conversion[species] else float("nan") \
                         for old_gene in deg_df["gene ID"]]
        else:
            new_genes = deg_df["gene ID"]
        
        # Add the orthogroups in an extra column
        deg_df["Orth_group"] = [gene2group_dict[gene] if gene in gene2group_dict else float("nan") for gene in new_genes]
        
        # Also convert the actual gene IDs
        deg_df["gene ID"] = new_genes
        
        # Remove NAs (due to no orthogroup, or no gene conversion)
        deg_df = deg_df.dropna(subset=["Orth_group", "gene ID"])
        
        # Select the cluster, gene and orthogroup columns and add to dict
        species2df_dict[species] = deg_df[["cluster", "gene ID", "Orth_group", "borda_rank"]]
    
    return species2df_dict


def get_correspondance_dict(correspondance_file, sep=","):
    
    """
    Get a dictionary of corresponding clusters in each species, based on a csv file.
    
    Parameters:
        correspondance_file (str): location of a csv file with a column "Tissue", and additional columns with the species 
                                   names, containing a comma-seprated list of cluster that are associated with that tissue 
                                   in each species.
                                   
    Output:
        correspondance_dict (dict): a tissue-to-species-to-clusters dictionary where the clusters are a list of strings
    """
    
    # Read in the csv file as a dataframe
    correspondance_df = pd.read_csv(correspondance_file, sep=sep, dtype=str, keep_default_na=False).set_index("Tissue")
    
    # Rearrange the dataframe into a tissue-to-species-to-clusters dict
    correspondance_dict = dict()
    for tissue in correspondance_df.index:
        for species in correspondance_df.columns:
            if tissue in correspondance_dict:
                correspondance_dict[tissue][species] = correspondance_df.loc[tissue, species].split(",")
            else:
                correspondance_dict[tissue] = {species: correspondance_df.loc[tissue, species].split(",")}
                
    return correspondance_dict


def convert_to_tissues(species2cluster2genes_and_groups, correspondance_dict):
    
    """
    Convert the clusters of different species into one tissue (i.e. cell type), merging the genes and orthogroups
    of different clusters of the same tissue together.
    
    Parameters:
        species2cluster2genes_and_groups (dict): A nested dictionary where the outer keys are species names, the middle keys 
                                                 are cluster names, and the innermost keys are "genes" and "groups" mapping 
                                                 to sets of genes and orthogroups respectively.
        correspondance_dict (dict): A dictionary mapping tissues to species and their corresponding clusters.
                                    The keys are tissue names, and values are dictionaries with species names as keys
                                    and lists of cluster names as values.

    Returns:
        dict: A nested dictionary where the outer keys are species names, the middle keys are tissue names, 
              and the innermost keys are "genes" and "groups" mapping to sets of genes and orthogroups respectively.
    
    """
    
    # Initialize the dictionary to store genes and orthogroups grouped by tissue
    species2tissue2genes_and_groups = dict()
    
    # Loop over all the species and tissues
    for species, cluster2genes_and_groups in species2cluster2genes_and_groups.items():
        species2tissue2genes_and_groups[species] = dict()
        for tissue in correspondance_dict:
            species2tissue2genes_and_groups[species][tissue] = dict()
            
            # Gather the genes or groups of all clusters of this tissue
            for genes_or_group in ["genes", "groups"]:
                id_set = set()
                for cluster in correspondance_dict[tissue][species]:
                    if (cluster == "") or (cluster not in cluster2genes_and_groups.keys()): 
                        pass # Skip if there is no cluster annotated to this tissue in this species
                    else: 
                        id_set.update(cluster2genes_and_groups[cluster][genes_or_group])
                
                # Add the merged gene or group IDs to the final dict
                species2tissue2genes_and_groups[species][tissue][genes_or_group] = id_set
                
    return species2tissue2genes_and_groups


def extract_cluster_groups(species2cluster2genes_and_groups, species_to_cluster):
    
    """
    Extract the orthogroups of a specific cluster for each species and prefix the cluster name to each group ID.

    Parameters:
        species2cluster2genes_and_groups (dict): A nested dictionary where the outer keys are species names, the middle keys 
                                                 are cluster names, and the innermost keys are "genes" and "groups" mapping 
                                                 to sets of genes and orthogroups respectively.
        species_to_cluster (dict): A dictionary mapping species names to specific cluster names to be extracted.

    Returns:
        dict: A dictionary where keys are species names and values are sets of orthogroups with the cluster name 
              prefixed to each orthogroup ID.
    """
    
    # Initialize the dictionary to store prefixed orthogroups for each species
    species_to_groups = dict()
    
    # Extract the orthogroups of a specific cluster in all species
    for species, cluster in species_to_cluster.items():
        
        # Add the cluster name as a prefix to the group ID
        species_to_groups[species] = {cluster+"_"+group for group in species2cluster2genes_and_groups[species][cluster]["groups"]}
        
    return species_to_groups


def extract_tissue_groups(species2cluster2genes_and_groups, correspondance_dict=None):
    
    """
    Extract orthogroups for each tissue across different species, combining clusters associated with the same tissue.

    Parameters:
        species2cluster2genes_and_groups (dict): A nested dictionary where the outer keys are species names, the middle keys 
                                                 are cluster names, and the innermost keys are "genes" and "groups" mapping 
                                                 to sets of genes and orthogroups respectively.
        correspondance_dict (dict, optional): A dictionary mapping tissues to species and their corresponding clusters.
                                              The keys are tissue names, and values are dictionaries with species names as keys
                                              and lists of cluster names as values. If None, the input dictionary 
                                              `species2cluster2genes_and_groups` is used directly without combining clusters.

    Returns:
        dict: A nested dictionary where the outer keys are tissue names and "all_tissues", 
              and values are dictionaries mapping species names to sets of orthogroups with prefixed cluster names.
    """
    
    # First combine the genes and groups of clusters of the same tissue if needed
    if correspondance_dict is not None:
        species2tissue2genes_and_groups = convert_to_tissues(species2cluster2genes_and_groups, correspondance_dict)
        tissues = set(correspondance_dict.keys())
    else:
        species2tissue2genes_and_groups = species2cluster2genes_and_groups
        tissues = set()
        for tissue_to_genes_and_groups in species2tissue2genes_and_groups.values():
            tissues.update(set(tissue_to_genes_and_groups.keys()))
    
    # Create a dictionary to store groups for all tissues combined
    species_to_groups_all = {species: set() for species in species2tissue2genes_and_groups.keys()}
    
    # Initialize the main dictionary to store groups by tissue
    tissue_to_species_to_groups = dict()
    
    for tissue in tissues:
        
        # Create a dummy dictionary mapping each species to the current tissue
        species_to_tissue_dummy = dict()
        for species in species2tissue2genes_and_groups.keys():
            species_to_tissue_dummy[species] = tissue
            
        # Extract the orthogroups of a specific tissue in all species
        tissue_to_species_to_groups[tissue] = extract_cluster_groups(species2tissue2genes_and_groups, species_to_tissue_dummy)
        
        # Add the species-specific groups to the dict of all tissues combined
        for species, groups in tissue_to_species_to_groups[tissue].items():
            species_to_groups_all[species].update(groups)
    
    # Add the combined groups to the main dictionary under "all_tissues"
    tissue_to_species_to_groups["all_tissues"] = species_to_groups_all
        
    return tissue_to_species_to_groups


def reverse_gene_to_group_dict(species2gene2group_dict):
    
    """
    Reverse the gene-to-group dictionary into a group-to-genes dictionary for each species.

    Parameters:
        species2gene2group_dict (dict): A dictionary with keys as species names and values as dictionaries.
                                        These inner dictionaries map gene IDs to orthogroup IDs.

    Returns:
        dict: A dictionary where the outer keys are species names and values are dictionaries.
              The inner dictionaries map orthogroup IDs to sets of gene IDs.
    """
    
    # Initialize the dictionary to store the reversed mappings
    species2group2genes_dict = dict()
    
    # Loop over each species and its gene-to-group dictionary
    for species, gene_to_group in species2gene2group_dict.items():
        species2group2genes_dict[species] = dict()
        for gene, group in gene_to_group.items():
            if group not in species2group2genes_dict[species].keys():
                species2group2genes_dict[species][group] = {gene}
            else:
                species2group2genes_dict[species][group].add(gene)
                
    return species2group2genes_dict


def get_top_N_orthogroups(df_best_degs, top_N):
    
    """
    Get the top N orthogroups from the dataframe.

    Parameters:
        df_best_degs (pd.DataFrame): A dataframe containing ranked orthogroups, with a column "Orth_group".
        top_N (int): The number of top orthogroups to include.

    Returns:
        pd.DataFrame: A dataframe slice containing the rows up to the index where the top N orthogroups are included.
                      If there are fewer than top N orthogroups, returns the entire dataframe.
    """
    top_orthogroups = set()
    top_N_orthogroup_idx = None
    
    # Iterate over the rows to collect top N orthogroups
    for i in range(df_best_degs.shape[0]):
        top_orthogroups.add(df_best_degs.loc[i, "Orth_group"])
        if len(top_orthogroups) == top_N:
            top_N_orthogroup_idx = i + 1
            break
    
    # Check if top N orthogroups were found, if not, return the entire dataframe
    if top_N_orthogroup_idx is None:
        top_N_orthogroup_idx = df_best_degs.shape[0]
    
    return df_best_degs.iloc[:top_N_orthogroup_idx].copy()


def extract_genes_and_orthogroups(species2df_dict, correspondance_dict, top_N=None):
    
    """
    Extract and organize genes and orthogroups for each tissue across different species, including top N orthogroups.

    Parameters:
        species2df_dict (dict): A dictionary with keys as species and values as dataframes containing
                                DEG information with columns "cluster", "gene ID", "Orth_group", "avg_log2FC", and "p_val_adj".
        correspondance_dict (dict): A dictionary mapping tissues to species and their corresponding clusters.
                                    The keys are tissue names, and values are dictionaries with species names as keys
                                    and lists of cluster names as values.
        top_N (int): The number of top orthogroups to include for each tissue.

    Returns:
        tuple: A tuple containing two dictionaries:
            - species2tissue2genes_and_groups: A dictionary with keys as species and values as dictionaries mapping tissue names 
                                               to sets of genes and orthogroups.
            - species2tissue2genes_and_groups_top: A dictionary with keys as species and values as dictionaries mapping tissue 
                                                   names to sets of genes and orthogroups, for the top N orthogroups.
    """

    # Initialize species-to-tissue-to-genes and orthogroups dict
    species2tissue2genes_and_groups = dict()

    for species, df in species2df_dict.items():
        
        species2tissue2genes_and_groups[species] = dict()
        for tissue, species_to_clusters in correspondance_dict.items():
            
            # For each tissue, only retain the best DEG if it is found in multiple clusters of that tissue
            clusters = species_to_clusters[species]
            if clusters:
                df_best_degs = df.query("cluster in @clusters")\
                                 .sort_values("borda_rank", ascending=True)\
                                 .drop_duplicates(subset="gene ID", keep="first")\
                                 .reset_index(drop=True)
                
                # If top N is specified, get the dataframe at which the top N orthogroups are included
                if top_N is not None:
                    df_best_degs = get_top_N_orthogroups(df_best_degs, top_N)
                    
                # Add the genes and orthogroups to the species-to-tissue-to-genes and orthogroups dict
                genes = set(df_best_degs["gene ID"])
                groups = set(df_best_degs["Orth_group"])
                species2tissue2genes_and_groups[species][tissue] = {"genes": genes, "groups": groups}
    
    return species2tissue2genes_and_groups


def extract_genes_and_orthogroups_degs(species2df_dict, correspondance_dict, top_N=None):
    
    """
    Extract and organize genes and orthogroups for each tissue across different species, including top N orthogroups.

    Parameters:
        species2df_dict (dict): A dictionary with keys as species and values as dataframes containing
                                DEG information with columns "cluster", "gene ID", "Orth_group", "avg_log2FC", and "p_val_adj".
        correspondance_dict (dict): A dictionary mapping tissues to species and their corresponding clusters.
                                    The keys are tissue names, and values are dictionaries with species names as keys
                                    and lists of cluster names as values.
        top_N (int): The number of top orthogroups to include for each tissue.

    Returns:
        tuple: A tuple containing two dictionaries:
            - species2tissue2genes_and_groups: A dictionary with keys as species and values as dictionaries mapping tissue names 
                                               to sets of genes and orthogroups.
            - species2tissue2genes_and_groups_top: A dictionary with keys as species and values as dictionaries mapping tissue 
                                                   names to sets of genes and orthogroups, for the top N orthogroups.
    """

    # Initialize species-to-tissue-to-genes and orthogroups dict
    species2tissue2genes_and_groups = dict()

    for species, df in species2df_dict.items():
        
        species2tissue2genes_and_groups[species] = dict()
        for tissue, species_to_clusters in correspondance_dict.items():
            
            # For each tissue, only retain the best DEG if it is found in multiple clusters of that tissue
            clusters = species_to_clusters[species]
            if clusters:
                df_best_degs = df.query("cluster in @clusters")\
                                 .sort_values(["p_val_adj", "avg_log2FC"], ascending=(True, False))\
                                 .drop_duplicates(subset="gene ID", keep="first")\
                                 .reset_index(drop=True)
                
                # If top N is specified, get the dataframe at which the top N orthogroups are included
                if top_N is not None:
                    df_best_degs = get_top_N_orthogroups(df_best_degs, top_N)
                    
                # Add the genes and orthogroups to the species-to-tissue-to-genes and orthogroups dict
                genes = set(df_best_degs["gene ID"])
                groups = set(df_best_degs["Orth_group"])
                species2tissue2genes_and_groups[species][tissue] = {"genes": genes, "groups": groups}
    
    return species2tissue2genes_and_groups


def extract_ranked_orthogroups(species2df_dict, correspondance_dict, top_N=np.inf):
    
    """
    Process DEG dataframes for multiple species and extract top N orthogroups for each tissue, including rank information.

    Parameters:
        species2df_dict (dict): A dictionary with keys as species and values as dataframes containing
                                DEG information with columns "cluster", "gene ID", "Orth_group", "avg_log2FC", and "p_val_adj".
        correspondance_dict (dict): A dictionary mapping tissues to species and their corresponding clusters.
                                    The keys are tissue names, and values are dictionaries with species names as keys
                                    and lists of cluster names as values.
        top_N (int): The number of top orthogroups to include for each tissue.

    Returns:
        dict: A nested dictionary where the outer keys are tissue names, the middle keys are species names,
              and the innermost values are dataframes containing the top N orthogroups with rank information.
    """
    
    # Initialize dict mapping tissues to species to a ranked DEG dataframe
    tissue_to_species_to_ranked_df = dict()

    for tissue, species_to_clusters in correspondance_dict.items():
        
        # Initialize dict mapping species to a ranked DEG dataframe
        tissue_to_species_to_ranked_df[tissue] = dict()
        
        for species, df in species2df_dict.items():
        
            # For each tissue, only retain the best DEG if it is found in multiple clusters of that tissue
            clusters = species_to_clusters[species]
            if clusters != [""]:
                df_best_degs = df.query("cluster in @clusters")\
                                 .sort_values("borda_rank", ascending=True)\
                                 .drop_duplicates(subset="gene ID", keep="first")\
                                 .reset_index(drop=True)
                
                # Get the dataframe at which the top N orthogroups are included
                df_best_degs_top = get_top_N_orthogroups(df_best_degs, top_N)
                
                # Calculate the ranks of that dataframe based on borda rank
                df_best_degs_top["Rank"] = list(df_best_degs_top.sort_values("borda_rank", ascending=True)\
                                                                .reset_index()\
                                                                .index+1
                                               )

                # Use the gene IDs as index
                df_best_degs_top = df_best_degs_top.set_index("gene ID", drop=False)
                
                # Add the dataframe with ranks to the final dict
                tissue_to_species_to_ranked_df[tissue][species] = df_best_degs_top
                
    return tissue_to_species_to_ranked_df
