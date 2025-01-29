"""
Script for running playlist continuation.
"""

import pandas as pd
import pickle
import pyxis as px
import sys
import argparse
import yaml
import os

sys.path.append(os.getcwd())

from playlist_cont_utils import get_playlist_cont_metrics, get_playlist_songs_sep
from model_utils import load_config
import torch

torch.set_num_threads(6)


def main(config):
    data_params = config['dataloader_params']['shared']
    split_ids = pickle.load(
        open(data_params['data_split_files'], "rb")
    )
    db_read = px.Reader(data_params['lmdb_cache_file'], lock=False)
    db_read_inds = pickle.load(open(data_params['lmdb_cache_inds_file'], "rb"))
    playlists = pd.concat(
        [
            pd.read_json("%s/%s.json" % (config['dataloader_params'][
                'audio_pair_files']['playlist_file_dir'], split))
            for split in ["train", "val", "test"]
        ]
    )

    # Separate playlists into training and testing
    playlist_songs_sep = get_playlist_songs_sep(split_ids, playlists)

    run_name =  str(config['wandb_params']["run_id"]) + "_" + config['wandb_params']['run_name']
    backbone_type = load_config(run_name, config['save_model_loc'])["model_params"]["audio"][
        "backbone_type"
    ]

    # Calculate playlist continuation metrics and print result
    metrics = get_playlist_cont_metrics(
        run_name,
        db_read,
        db_read_inds,
        backbone_type,
        split_ids,
        playlist_songs_sep,
        resnet_batches=550, # Update batch sizes based on available GPU memory
        sc_cnn_batches=2400,
        k=100, # top k for metric calculation
        sim_batch_size=15000,
        save_model_loc=config['save_model_loc'],
        devices=[0],
    )
    recall, m_ap, ndcg = metrics
    print('Recall:',recall,'MAP:',m_ap,'NDCG:',ndcg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append")

    config_file = parser.parse_args().config

    with open(config_file[0], "r") as in_f:
        config = yaml.safe_load(in_f)

    main(config)

    
    

