"""
Utility functions for playlist continuation downstream task.
"""

import pandas as pd
import torch
import numpy as np
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import ndcg_score
import os
sys.path.append(os.getcwd())
from model_utils import get_backbone
import pickle

def process_batch(batch, db_read):
    """
    Loads batch spectrograms from Melon LMDB cache.
    """
    specs_all = []
    files = []
    for i, spec_ind in enumerate(batch[0]):
        spec = db_read[spec_ind]["spec"]
        spec_sliced = get_spec_slices(spec)
        slice_file_map = [spec_ind] * len(spec_sliced)

        specs_all += spec_sliced
        files += slice_file_map
    spec_batch_input = np.stack(specs_all)
    return spec_batch_input, np.array(files)


def get_spec_slices(spec, slice_size=256):
    """
    Gets all non-overlapping patches from spectrogram.
    """
    if spec.shape[1] < slice_size:
        spec = np.pad(spec, ((0, 0), (0, slice_size - spec.shape[1])))
    num_slices = max(spec.shape[1] // slice_size, 1)

    spec_split = np.array_split(spec[:, : slice_size * num_slices], num_slices, axis=1)
    return spec_split


def generate_sparse(argsort, k, num_vad_songs):
    """
    Get sparse binary matrix indicating which songs belong to which playlists.
    """
    argsort_k = argsort[:, :k]
    data = [1 for i in range(argsort_k.shape[0] * k)]
    row_ind = [i for i in range(argsort_k.shape[0]) for _ in range(k)]
    col_ind = [x for y in argsort_k for x in y]
    return sp.csr_matrix(
        (data, (row_ind, col_ind)), shape=(argsort_k.shape[0], num_vad_songs)
    )


def average_precision_k(target, predicted, k=100):
    """
    Computes the average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    hits = 0
    val = 0

    for i, p in enumerate(predicted):
        if p in target and p not in predicted[:i]:
            hits += 1
            val += hits / (i + 1)

    ap = val / min(len(target), k)
    return ap


def get_model_outputs(
    run_name,
    save_model_loc,
    read_inds_batched,
    db_read,
    db_read_inds,
    devices=[0],
):
    """
    Gets audio encoder outputs for all songs in Melon data, using similar data
    loading loop to the trainers.
    """
    model = get_backbone(run_name, save_model_loc).to(torch.device("cuda"))

    model.eval()
    model = torch.nn.DataParallel(model, device_ids=devices)
    dl = iter(
        DataLoader(
            read_inds_batched,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            prefetch_factor=1,
            drop_last=False,
            collate_fn=lambda batch: process_batch(batch, db_read),
        )
    )
    vecs_dict = {}
    for i in range(len(dl)):
        batch = read_inds_batched[i]
        spec_batch_input, file_arr = next(dl)
        inp = torch.from_numpy(spec_batch_input).to(torch.device("cuda"))
        with torch.no_grad():
            model_out = model(inp)
        model_out = model_out.cpu().detach()
        for file in batch:
            inds = np.where(file_arr == file)[0]
            vecs_dict[db_read_inds[file]] = model_out[inds].mean(axis=0).numpy()

        torch.cuda.empty_cache()
    return vecs_dict


def get_playlist_songs_sep(split_ids, playlists):
    """
    Separates all playlists into train/validation and test songs.
    """
    train_ids, val_ids, test_ids = split_ids
    playlist_songs = playlists["songs"].values
    train_set = set(train_ids).union(set(val_ids))
    test_set = set(test_ids)
    playlist_songs_sep = []
    for songs in playlist_songs:
        train_ = []
        test_ = []
        for song in songs:
            if song in train_set:
                train_.append(song)
            elif song in test_set:
                test_.append(song)
        if len(test_) > 0 and len(train_) > 0:
            playlist_songs_sep.append((train_, test_))
    return playlist_songs_sep


def cosine_similarity_matrix(z, y):
    """Calculates cosine similarity matrix between two embedding tensors."""
    cosine_similarity = torch.matmul(z, y.t())
    embedding_norms_z = torch.norm(z, p=2, dim=1)
    embedding_norms_y = torch.norm(y, p=2, dim=1)
    embedding_norms_mat = embedding_norms_y.unsqueeze(0) * embedding_norms_z.unsqueeze(
        1
    )
    cosine_similarity = cosine_similarity / torch.maximum(
        embedding_norms_mat, torch.tensor(1e-8)
    )
    return cosine_similarity


def get_sim_data(
    vecs_dict, playlist_songs_sep, full_ids, test_ids, k=100, batch_size=19500
):
    """
    Calculates top k most similar songs in test set for all training/validation songs.
    Similarities are calculated in batches to prevent memory issues.

    Args:
        vecs_dict: Dictionary containing audio encoder output for all songs
        playlist_songs_sep: Melon playlists separated into train/validation and test
        full_ids: Train/validation set song IDs
        test_ids: Test set song IDs
        k: Number of most similar songs calculated for each song
        batch_size: Number of train/validation songs to include in each similarity batch

    Returns:

    """
    full_arr_tensor = torch.from_numpy(np.stack([vecs_dict[k] for k in full_ids]))
    test_arr_tensor = torch.from_numpy(np.stack([vecs_dict[k] for k in test_ids]))
    full_id_playlist_map = {ind: [] for ind in full_ids}
    playlist_train_len_scalars = [1 / len(p[0]) for p in playlist_songs_sep]

    for ind, (pl_train, pl_test) in enumerate(playlist_songs_sep):
        for full_id in pl_train:
            full_id_playlist_map[full_id].append(ind)

    # Initialise matrix for tracking overall similarities for test songs to each playlist
    # (to be used as a tiebreaker)
    playlist_test_scores = np.zeros((len(playlist_songs_sep), len(test_ids))).astype(
        np.float32
    )
    batches = []
    num_batches = len(full_ids) // batch_size + 1
    for i in range(num_batches):
        sim = cosine_similarity_matrix(
            full_arr_tensor[batch_size * i : batch_size * (i + 1)], test_arr_tensor
        ).numpy()

        # Partition top k most similar songs, get values, and sort to get correct order
        sim_partitioned = np.argpartition(-sim, k)[:, :k].copy()
        sim_partitioned_vals = np.take_along_axis(-sim, sim_partitioned, axis=-1)
        sim_partitioned_argsort = np.argsort(sim_partitioned_vals)
        sim_argsort = np.take_along_axis(
            sim_partitioned, sim_partitioned_argsort, axis=-1
        )
        batches.append(sim_argsort)

        batch_full_ids = full_ids[batch_size * i : batch_size * (i + 1)]

        # Update playlist scores
        for j, full_id in enumerate(batch_full_ids):
            playlist_test_scores[full_id_playlist_map[full_id]] += sim[j]

    sim_argsort_concat = np.concatenate(batches)

    # Divide playlist scores by length to get average
    playlist_test_scores *= np.array(np.expand_dims(playlist_train_len_scalars, axis=1))
    return sim_argsort_concat, playlist_test_scores


def get_top_k(
    playlist_songs_sep,
    playlist_test_scores,
    sim_argsort_concat,
    full_id_map,
    i,
    k=100,
):
    """
    Given similarities, calculate top k for each playlist, using number of
    appearances across playlist first then average overall similarity as a
    tiebreaker.
    """
    pl_train, pl_test = playlist_songs_sep[i]

    avg_sims = playlist_test_scores[i]
    sim_argsort = sim_argsort_concat[[full_id_map[j] for j in pl_train]]

    sim_argsort_flat = np.reshape(sim_argsort, (-1))
    unique_full, counts_full = np.unique(sim_argsort_flat, return_counts=True)
    k = min([k, len(counts_full) - 1])
    lowest_count = -np.sort(-counts_full)[k]
    unique = unique_full[counts_full >= lowest_count]
    counts = counts_full[counts_full >= lowest_count]
    top_avg_sims = avg_sims[unique]

    top_both_arr = np.stack([counts, top_avg_sims], axis=1)
    top_both = np.empty(len(top_both_arr), dtype=object)
    for i in range(len(top_both_arr)):
        top_both[i] = tuple(-top_both_arr[i])
    top_argpartition = np.argpartition(top_both, axis=0, kth=k)[:k]
    top_vals = top_both[top_argpartition]
    top_argsort = np.argsort(top_vals)
    return unique[top_argpartition[top_argsort]]


def get_metrics(
    playlist_test_scores,
    sim_argsort_concat,
    playlist_songs_sep,
    test_id_map,
    full_id_map,
    k,
):
    """
    Calculate playlist continuation metrics given top k.
    """
    metrics = []
    for i, (pl_train, pl_test) in enumerate(playlist_songs_sep):
        pl_test_ind = [test_id_map[song] for song in pl_test]
        ot = get_top_k(
            playlist_songs_sep,
            playlist_test_scores,
            sim_argsort_concat,
            full_id_map,
            i,
            k=k,
        )
        ap = average_precision_k(pl_test_ind, ot, k)
        ot_correct_oh = [int(x in pl_test_ind) for x in ot]
        ndcg = ndcg_score([ot_correct_oh], [list(range(len(ot), 0, -1))], k=len(ot))
        ot_correct = sum(ot_correct_oh)
        metrics.append((ot_correct / len(pl_test_ind), ap, ndcg))
    return np.mean(np.array(metrics), axis=0)


def get_playlist_cont_metrics(
    run_name,
    db_read,
    db_read_inds,
    backbone_type,
    split_ids,
    playlist_songs_sep,
    resnet_batches=500,
    sc_cnn_batches=2400,
    k=100,
    sim_batch_size=19500,
    save_model_loc="./runs",
    devices=[0],
):
    """
    Main function for calculating playlist continuation metrics.
    """
    if backbone_type == "resnet":
        num_batches = resnet_batches
    elif backbone_type == "sc_cnn":
        num_batches = sc_cnn_batches
    read_inds_batched = np.array_split(list(range(len(db_read))), num_batches)
    vecs_dict = get_model_outputs(
        run_name,
        save_model_loc,
        read_inds_batched,
        db_read,
        db_read_inds,
        devices=devices,
    )
    pickle.dump(
        vecs_dict,
        open(
            "/homes/gm729/melon_contrastive/gm_final/data/rec_testing_dict_new.pkl", "wb"
        ),
    )
    train_ids, val_ids, test_ids = split_ids
    test_id_map = {k: i for i, k in enumerate(test_ids)}
    vecs_set = set(list(vecs_dict))
    full_ids = [x for x in train_ids + val_ids if x in vecs_set]
    full_id_map = {k: i for i, k in enumerate(full_ids)}

    sim_argsort_concat, playlist_test_scores = get_sim_data(
        vecs_dict,
        playlist_songs_sep,
        full_ids,
        test_ids,
        k=k,
        batch_size=sim_batch_size,
    )

    metrics = get_metrics(
        playlist_test_scores,
        sim_argsort_concat,
        playlist_songs_sep,
        test_id_map,
        full_id_map,
        k,
    )
    return metrics
