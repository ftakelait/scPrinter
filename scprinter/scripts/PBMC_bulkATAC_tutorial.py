import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="PBMC bulk ATAC tutorial")
    parser.add_argument("--main-dir", default="./PBMC_bulkATAC_tutorial", help="Base directory for outputs")
    parser.add_argument(
        "--samples",
        default="Bcell_0,Bcell_1,Monocyte_0,Monocyte_1,Tcell_0,Tcell_1",
        help="Comma-separated sample names",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold index for training")
    args = parser.parse_args()

    
    
    import scprinter as scp
    
    import pandas as pd
    
    import seaborn as sns
    
    import matplotlib.pyplot as plt
    
    import time
    
    import pandas as pd
    
    import numpy as np
    
    import os
    
    import pickle
    
    import torch
    
    import matplotlib as mpl
    
    mpl.rcParams['pdf.fonttype'] = 42
    
    from scanpy.plotting.palettes import zeileis_28
    
    from tqdm.contrib.concurrent import *
    
    from tqdm.auto import *
    
    import anndata
    
    import scanpy as sc
    
    import json
    
    import csv
    
    import re
    
    from sklearn.preprocessing import OneHotEncoder
    main_dir = args.main_dir
    
    work_dir = f'{main_dir}/seq2print'
    os.makedirs(work_dir, exist_ok=True)
    
    frag_dir = f'{main_dir}/fragments'
    os.makedirs(frag_dir, exist_ok=True)
    # Download the fragments files of bulk samples from our Zenodo repository https://zenodo.org/records/14866808 using the code below
    
    samples = args.samples.split(",")
    
    for sample in samples:
    
        link_prefix = "https://zenodo.org/records/14866808/files/PBMC_bulk_ATAC_tutorial_"
    
        link_suffix = "_frags.tsv.gz?download=1"
    
        subprocess.run(
            ["wget", "-O", f"{frag_dir}/{sample}_frags.tsv.gz", f"{link_prefix}{sample}{link_suffix}"],
            check=True,
        )
    # Get the fragments files for individual samples
    
    # Note: the fragments files should have 4 columns: chromosome, start, end, and barcode. 
    
    # Barcode should just be sample name, which means all fragments in the same fragments file have the same barcode.
    
    frag_files = os.listdir(frag_dir)
    
    frag_files = [i for i in frag_files if re.search("frags.tsv.gz", i) is not None]
    
    frag_files = sorted([os.path.join(frag_dir, i) for i in frag_files])
    
    samples = ["_".join(re.split("[/_\\.]", i)[11:13]) for i in frag_files]
    
    samples
    printer = scp.pp.import_fragments(
    
                            path_to_frags=frag_files,
    
                            barcodes=[None] * len(frag_files), # This loads individual fragments files and combine them
    
                            savename=os.path.join(work_dir, 'PBMC_bulkATAC_scprinter.h5ad'),
    
                            genome=scp.genome.hg38,
    
                            min_num_fragments=1000, min_tsse=7,
    
                            sorted_by_barcode=False, 
    
                            low_memory=False,
    
                            )
    print (printer.insertion_file.obs_names[:])
    # Rename barcodes to sample IDs if needed. 
    
    printer.insertion_file.obs_names = samples
    # Call peaks, this set of peaks are recommended to train seq2PRINT model
    
    scp.pp.call_peaks(printer=printer,
    
                      frag_file=frag_files,
    
                      cell_grouping=[None], # here we call peaks on the cells that are included in the final analyses
    
                      group_names=['all'],
    
                      preset='seq2PRINT',
    
                      overwrite=False)
    
    
    
    # Fetched the cleaned peaks, save, it will be used in the next step
    
    cleaned_peaks = pd.DataFrame(printer.uns["peak_calling"]['all_cleaned'][:])
    
    cleaned_peaks.to_csv(f'{work_dir}/seq2print_cleaned_narrowPeak.bed', 
    
                         sep='\t', header=False, index=False)
    
    # Call peaks using chromvar preset, this set of peak are recommended to be use as cell x peak for scATAC-seq data, or analysis
    
    scp.pp.call_peaks(printer=printer,
    
                      frag_file=frag_files,
    
                      cell_grouping=[None], # here we call peaks on the cells that are included in the final analyses
    
                      group_names=['chromvar_all'],
    
                      preset='chromvar',
    
                      overwrite=False)
    
    
    
    # Fetched the cleaned peaks, save, it will be used in the next step
    
    cleaned_peaks = pd.DataFrame(printer.uns["peak_calling"]['chromvar_all_cleaned'][:])
    
    cleaned_peaks.to_csv(f'{work_dir}/regions.bed', 
    
                         sep='\t', header=False, index=False)
    # we can compare two sets of peaks with different preset
    
    print (pd.DataFrame(printer.uns["peak_calling"]['all_cleaned'][:]))
    print (pd.DataFrame(printer.uns["peak_calling"]['chromvar_all_cleaned'][:]))
    import json
    
    model_configs = []
    
    if not os.path.exists(os.path.join(work_dir, 'configs')):
    
        os.makedirs(os.path.join(work_dir, 'configs'))
    
    for sample in samples:
    
        # For exploratory analyses we run only one fold. For publication results you can run all 5 fold for slight improvement in accuracy.
    
        # If you want to run all five folds, add a for-loop layer "for fold in range(5)". This way each sample will have 5 models trained.
    
        fold = args.fold
    
        model_config= scp.tl.seq_model_config(printer,
    
                                         region_path=f'{work_dir}/seq2print_cleaned_narrowPeak.bed',
    
                                         cell_grouping=[sample],
    
                                         group_names=sample,
    
                                         genome=printer.genome,
    
                                           fold=args.fold,
    
                                         overwrite_bigwig=False,
    
                                         model_name='PBMC_bulkATAC_' + sample,
    
                                         additional_config={
    
                                            "notes": "v3",
    
                                            "tags": ["PBMC_bulkATAC", sample, 
    
                                                f"fold{fold}"]},
    
                                         path_swap=(work_dir, ''),
    
                                         config_save_path=f'{work_dir}/configs/PBMC_bulkATAC_{sample}_fold{fold}.JSON')
    
        model_configs.append(model_config)
    for path in ['temp','model']:
    
        if not os.path.exists(os.path.join(work_dir, path)):
    
            os.makedirs(os.path.join(work_dir, path))
    
    
    
    for sample in samples:
    
        scp.tl.launch_seq2print(model_config_path=f'{work_dir}/configs/PBMC_bulkATAC_{sample}_fold{fold}.JSON',
    
                                temp_dir=f'{work_dir}/temp',
    
                                model_dir=f'{work_dir}/model',
    
                                data_dir=work_dir,
    
                                gpus=samples.index(sample),
    
                                wandb_project='scPrinter_seq_PBMC_bulkATAC', # wandb helps you manage loggins
    
                                verbose=False,
    
                                launch=False # launch=True, this command would launch the scripts directly,
    
                                # otherwise, it will just display the commands, you should copy them and run them.
    
                               )
    # Now fetch models:
    
    import wandb
    
    
    
    # Login to your W&B account
    
    wandb.login()
    
    
    
    # Set your entity and project
    
    entity = 'ruochiz'  # Replace with your W&B entity (username or team name)
    
    project = 'scPrinter_seq_PBMC_bulkATAC'  # Replace with your W&B project name
    
    
    
    # Initialize the API
    
    api = wandb.Api()
    
    
    
    # Get the project
    
    runs = api.runs(f"{entity}/{project}")
    
    
    
    pretrain_models = []
    
    model_path_dict = {}
    
    for run in runs:
    
        if run.state != 'finished':
    
            continue
    
        if 'PBMC_bulkATAC' in run.tags:
    
            sample = [i for i in run.tags if i in samples][0]
    
            model_name = run.config['savename'] + '-' + run.name + '.pt'
    
            model_path = os.path.join(work_dir, "model", model_name)
    
            model_path_dict[sample] = model_path
    model_path_dict
    regions_dict = {
    
        'chr11:47378230-47379030': 'SPI1', # Highly expressed in monocytes and B cells
    
        'chr5:140633035-140633835': 'CD14', # Monocyte marker
    
        'chr19:41876833-41877633': 'CD79a', # B cell marker
    
        'chr11:118342294-118343094': 'CD3D', # T cell marker
    
        'chr17:58281312-58282112': 'MPO', # Monocyte marker
    
        'chr16:31259594-31260394': 'ITGAM', # Expressed in myeloid cells (including monocyte)
    
        'chr3:39281281-39282081': 'CX3CR1', # A pretty important gene driving disease-related monocyte cell states
    
    }
    
    
    
    # Save example regions to a bed file
    
    regions_df = []
    
    for region in regions_dict:
    
        regions_df.append(re.split("[:-]", region))
    
    regions_df = pd.DataFrame(regions_df)
    
    regions_df.to_csv(f'{work_dir}/regions_test.bed',
    
                      sep='\t', header=False, index=False)
    samples
    import json
    
    adata_tfbs = {}
    
    for sample_ind, sample in enumerate(samples):
    
        adata_tfbs[sample] = scp.tl.seq_tfbs_seq2print(seq_attr_count=None,
    
                              seq_attr_footprint=None,
    
                              genome=printer.genome,
    
                              region_path=f'{work_dir}/regions_test.bed',
    
                              gpus=[5,6,7], # change it to the available gpus
    
                              model_type='seq2print',
    
                              model_path=model_path_dict[sample], # For now we just run on one fold but you can provide a list of paths to all 5 folds
    
                              lora_config=json.load(open(f'{work_dir}/configs/PBMC_bulkATAC_{sample}_fold0.JSON', 'r')),
    
                              group_names=[sample],
    
                              verbose=False,
    
                              launch=True,
    
                              return_adata=True, # turn this as True
    
                              overwrite_seqattr=True,
    
                              post_normalize=False,
    
                              save_key=f'PBMC_bulkATAC_{sample}_roi', # and input a save_key
    
                              save_path=work_dir)
    if not os.path.exists(f'{work_dir}/plots'):
        os.makedirs(f'{work_dir}/plots', exist_ok=True)
    
    for region in regions_dict.keys():
    
        print(region, regions_dict[region])
    
        tfbs = pd.DataFrame(np.array([adata_tfbs[sample].obsm[region] for sample in samples]).squeeze(), index=samples)
    
        sns.heatmap(tfbs, cmap='RdBu_r')
    
        plt.savefig(f'{work_dir}/plots/TFBS_{region}.png')
    
        plt.show()
    # First construct a peak-by-cell matrix of ATAC counts
    
    peak_path = f'{work_dir}/regions.bed'
    
    adata = scp.pp.make_peak_matrix(printer,
    
                           regions=peak_path,
    
                           region_width=300,
    
                           cell_grouping=None,
    
                           group_names=None,
    
                           sparse=True)
    
    adata.write(f'{work_dir}/cell_peak.h5ad')
    # Remove regions with low coverage (helps to reduce total peak number and save time. The full list of 300k peaks contains many very weak peaks)
    
    regions = pd.read_csv(f'{work_dir}/regions.bed', sep='\t', header=None)
    
    adata = anndata.read_h5ad(f'{work_dir}/cell_peak.h5ad')
    
    peak_depth = np.array(np.sum(adata.X, axis=0)).squeeze()
    
    regions_filt = regions.iloc[np.where(peak_depth > 200)[0], :]
    
    regions_filt.to_csv(f'{work_dir}/regions_filt.bed', 
    
                         sep='\t', header=False, index=False)
    import json
    
    for sample_ind, sample in enumerate(samples):
    
        scp.tl.seq_tfbs_seq2print(seq_attr_count=None,
    
                              seq_attr_footprint=None,
    
                              genome=printer.genome,
    
                              region_path=f'{work_dir}/regions_filt.bed',
    
                              gpus=[5,6,7],
    
                              model_type='seq2print',
    
                              model_path=model_path_dict[sample], # For now we just run on one fold
    
                              lora_config=json.load(open(f'{work_dir}/configs/PBMC_bulkATAC_{sample}_fold{fold}.JSON', 'r')),
    
                              group_names=[sample],
    
                              verbose=True,
    
                              launch=True,
    
                              return_adata=False, 
    
                              overwrite_seqattr=True,
    
                              post_normalize=True,
    
                              save_key=f'PBMC_bulkATAC_{sample}', # and input a save_key
    
                              save_path=work_dir)
    # We first scan TF motifs across all regions to find motif matched sites
    
    
    
    # Initialize motif set object
    
    motifs = scp.motifs.FigR_Human_Motifs(genome=printer.genome, bg=[0.25] * 4)
    
    
    
    # Prepare motif scanner. Here you can specify which TF motifs you want to scan using tf_genes. If tf_genes=None then use all motifs
    
    motifs.prep_scanner()
    
    
    
    # Scan motif sites. This will return the exact genomic coordinates of motif matches
    
    motif_sites = motifs.scan_motif(regions_filt, verbose=True, clean=True)
    
    
    
    # Reformat motif matches to a pandas DataFrame
    
    motif_sites = pd.DataFrame(motif_sites)
    
    motif_sites.iloc[:, 2] = motif_sites.iloc[:, 1] + motif_sites.iloc[:, 8]
    
    motif_sites.iloc[:, 1] = motif_sites.iloc[:, 1] + motif_sites.iloc[:, 7]
    
    motif_sites = motif_sites.iloc[:, [0,1,2,4]]
    
    motif_sites.columns=["chrom", "start", "end", "TF"]
    # We then extract the TF binding scores at those motif sites
    
    def fetch_bw(args):
    
        import pyBigWig as pw
    
        
    
        TFBS, bw, genome = args
    
        chroms, starts, ends = np.array(TFBS['chrom']),np.array(TFBS['start']),np.array(TFBS['end'])
    
        res_all = {}
    
        with pw.open(bw, 'r') as f:
    
            for chrom in tqdm(genome.chrom_sizes):
    
                if chrom == 'chrY':
    
                    continue
    
                res_all[chrom] = f.values(chrom, 0, genome.chrom_sizes[chrom], numpy=True)
    
            
    
        vs = []
    
        for chr, left, right in zip(tqdm(chroms, mininterval=1), starts, ends):
    
            vs.append(np.nanmean(res_all[chr][left:right]))
    
        return vs
    
    
    
    # Multi-process loading of TF binding scores 
    
    bigwig_dict = {sample:f"{work_dir}/{sample}_TFBS.bigwig" for sample in samples}
    
    args = [[motif_sites, bigwig_dict[sample], printer.genome] for sample in samples]
    
    n_jobs = 4
    
    import multiprocessing as mp
    
    with mp.Pool(n_jobs) as pool:
    
        TFBS_scores = list(pool.imap(fetch_bw, args))
    
    TFBS_scores = np.array(TFBS_scores).T
    
    TFBS_scores = pd.DataFrame(TFBS_scores, columns=[f"TFBS_{sample}" for sample in samples])
    
    TFBS_scores = pd.concat([motif_sites, TFBS_scores], axis=1)
    TFBS_scores.head()
    fig, ax = plt.subplots(2,2, figsize=(7,6))
    
    
    
    scores = TFBS_scores.loc[TFBS_scores.TF.values == "CEBPA", :]
    
    ax[0][0].scatter(
    
        np.mean(scores.loc[:, [f"TFBS_Monocyte_{i}" for i in range(2)]], axis=1), 
    
        np.mean(scores.loc[:, [f"TFBS_Tcell_{i}" for i in range(2)]], axis=1), s=0.01)
    
    ax[0][0].set_xlabel("Monocyte TFBS")
    
    ax[0][0].set_ylabel("T cell TFBS")
    
    ax[0][0].set_title("CEBPA")
    
    plt.tight_layout()
    
    
    
    scores = TFBS_scores.loc[TFBS_scores.TF.values == "SPI1", :]
    
    ax[0][1].scatter(
    
        np.mean(scores.loc[:, [f"TFBS_Tcell_{i}" for i in range(2)]], axis=1), 
    
        np.mean(scores.loc[:, [f"TFBS_Bcell_{i}" for i in range(2)]], axis=1), s=0.01)
    
    ax[0][1].set_xlabel("T cell TFBS")
    
    ax[0][1].set_ylabel("B cell TFBS")
    
    ax[0][1].set_title("SPI1")
    
    plt.tight_layout()
    
    
    
    scores = TFBS_scores.loc[TFBS_scores.TF.values == "PAX5", :]
    
    ax[1][0].scatter(
    
        np.mean(scores.loc[:, [f"TFBS_Tcell_{i}" for i in range(2)]], axis=1), 
    
        np.mean(scores.loc[:, [f"TFBS_Bcell_{i}" for i in range(2)]], axis=1), s=0.01)
    
    ax[1][0].set_xlabel("T cell TFBS")
    
    ax[1][0].set_ylabel("B cell TFBS")
    
    ax[1][0].set_title("PAX5")
    
    plt.tight_layout()
    
    
    
    scores = TFBS_scores.loc[TFBS_scores.TF.values == "RUNX3", :]
    
    ax[1][1].scatter(
    
        np.mean(scores.loc[:, [f"TFBS_Tcell_{i}" for i in range(2)]], axis=1), 
    
        np.mean(scores.loc[:, [f"TFBS_Bcell_{i}" for i in range(2)]], axis=1), s=0.01)
    
    ax[1][1].set_xlabel("T cell TFBS")
    
    ax[1][1].set_ylabel("B cell TFBS")
    
    ax[1][1].set_title("RUNX3")
    
    
    
    plt.show()
    scores = TFBS_scores.loc[TFBS_scores.TF.values == "RUNX3", :]
    
    Tcell_scores = np.mean(scores.loc[:, [f"TFBS_Tcell_{i}" for i in range(2)]], axis=1)
    
    Bcell_scores = np.mean(scores.loc[:, [f"TFBS_Bcell_{i}" for i in range(2)]], axis=1)
    
    diff = Tcell_scores - Bcell_scores
    
    diff = diff[np.abs(diff) > 0.05]
    
    plt.hist(diff, bins=100)
    
    plt.show()
    printer.close()


if __name__ == "__main__":
    main()
