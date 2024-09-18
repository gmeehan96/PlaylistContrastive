"""
This script creates the train/validation/test split files for the Melon data and the 
Melon-50 subgenre task. It also generates the collaborative filtering vectors used in some 
training configurations. Outputs are saved as pickle files in the ../data folder.


Some functions are taken or adapted from the following script in the
contrastive-mir-learning repository (corresponding to reference [18] in the paper):
https://github.com/andrebola/contrastive-mir-learning/blob/master/scripts/create_dataset.py
"""

import sys
sys.path.append(".")

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy import sparse
from lightfm import LightFM
from data_loading import generate_pairs



# CF vector parameters
CF_DIMS = 300
CF_EPOCHS = 50
CF_MAX_SAMPLED = 40
CF_LR = 0.05

# File locations
META_DIR = (
    ".../kakao_meta"  # UPDATE
)
TRAIN_FILE = "%s/train.json" % META_DIR
CACHE_DIR = "./data/Melon"

# Genre map file from contrastive-mir-learning repository
GENRE_MAP_FILE = "./data/w2v_all_gnr_map.json"

# write output files to
SAVE_DATASET_LOCATION = "./data/"


def train_mf(impl_train_data, dims=200, epochs=50, max_sampled=10, lr=0.05):
    """
    Trains collaborative filtering model using WARP loss.
    """

    model = LightFM(
        loss="warp",
        no_components=dims,
        max_sampled=max_sampled,
        learning_rate=lr,
        random_state=42,
    )
    model = model.fit(impl_train_data, epochs=epochs, num_threads=24)

    item_biases, item_embeddings = model.get_item_representations()
    item_vec = np.concatenate(
        (item_embeddings, np.reshape(item_biases, (1, -1)).T), axis=1
    )
    return item_vec


def load_cf_data(train_file, tracks_ids):
    """
    Gets binary matrix for CF vectors from playlist data and performs training.
    """

    train_playlists = json.load(open(train_file, encoding="utf-8"))

    rows = []
    cols = []
    data = []
    playlists_ids = []
    playlists_test = {}
    for playlist in train_playlists:
        for track in playlist["songs"]:
            if track in tracks_ids:
                cols.append(tracks_ids[track])
                rows.append(len(playlists_ids))
                data.append(1)
            else:
                if str(playlist["id"]) not in playlists_test:
                    playlists_test[str(playlist["id"])] = []
                playlists_test[str(playlist["id"])].append(str(track))
        playlists_ids.append(playlist["id"])
    train_coo = sparse.coo_matrix((data, (rows, cols)), dtype=np.float32)

    item_vec = train_mf(
        train_coo, dims=CF_DIMS, epochs=CF_EPOCHS, max_sampled=CF_MAX_SAMPLED, lr=CF_LR
    )
    return item_vec


def cf_wrapper(sound_tags_num, split_info):
    """
    Wrapper function for producing CF vectors.
    """

    inv_sound_tags_num = {v: k for k, v in sound_tags_num.items()}
    print("Creating CF vectors")
    item_vec = load_cf_data(TRAIN_FILE, inv_sound_tags_num)
    train_cf_vec = item_vec[split_info[0]]
    val_cf_vec = item_vec[split_info[1]]
    return train_cf_vec, val_cf_vec


def get_sound_tags(song_meta, genre_w2v_map):
    """
    Utility function for collecting genre and sub-genre tags from song metadata
    and mapping them to indices.
    """

    song_meta["w2v"] = song_meta.apply(
        lambda row: [
            genre_w2v_map[g]
            for g in row["song_gn_dtl_gnr_basket"] + row["song_gn_gnr_basket"]
        ],
        axis=1,
    )
    sound_tags = song_meta.set_index("id")["w2v"].to_dict()
    return sound_tags


def get_data_split(sound_tags):
    """
    Calculates train/validation/test split for Melon data using stratification by
    genre, and runs CF vector training for the train and validation songs.

    Args:
        sound_tags: dictionary containing list of genre tags for each song

    Returns:
        Tuple containing song ids for each split
        Tuple containing train and validation CF vectors
    """
    sound_tags_num = {i: k for i, k in enumerate(sound_tags.keys())}

    # Perform first stratified split to get test set
    mlb = MultiLabelBinarizer()
    binarized_gnr = mlb.fit_transform(list(sound_tags.values()))
    print("Dimensions", binarized_gnr.shape)
    msss = MultilabelStratifiedShuffleSplit(test_size=0.1, random_state=0, n_splits=1)
    r = list(msss.split(list(sound_tags_num.keys()), binarized_gnr))
    train_ids_tmp = [sound_tags_num[i] for i in r[0][0]]
    test_ids = [sound_tags_num[i] for i in r[0][1]]

    # Perform second stratified split to get train and validation split
    sound_tags_num = {i: k for i, k in enumerate(train_ids_tmp)}
    binarized_gnr = mlb.fit_transform([sound_tags[k] for k in train_ids_tmp])
    msss = MultilabelStratifiedShuffleSplit(test_size=1 / 9, random_state=0, n_splits=1)
    r = list(msss.split(list(sound_tags_num.keys()), binarized_gnr))
    train_ids = [sound_tags_num[i] for i in r[0][0]]
    val_ids = [sound_tags_num[i] for i in r[0][1]]

    # Get CF vectors
    train_cf_vec, val_cf_vec = cf_wrapper(sound_tags_num, r[0])

    num_training_instances = len(train_ids)
    num_validation_instances = len(val_ids)
    num_test_instances = len(test_ids)

    print("Num training instances: {}".format(num_training_instances))
    print("Num validation instances: {}".format(num_validation_instances))
    print("Num test instances: {}".format(num_test_instances))
    return (train_ids, val_ids, test_ids), (train_cf_vec, val_cf_vec)


def get_melon_sg_split(test_ids, song_meta, db_inds):
    """
    Calculates top 50 sub-genre tags and train/validation/test split for the Melon-50
    tagging task, and runs CF vector training for the train and validation songs.

    Args:
        test_ids: IDs of songs in test sets
        song_meta: Song metadata dataframe from Melon data
        db_inds: indices of songs in Melon LMDB

    Returns:
        melon_annotations: dataframe of top-50 labels for songs in sub-genre task
        sg_split: Tuple containing train/validation/test song IDs for sub-genre task
    """
    db_read_inds_map = {k: i for i, k in enumerate(db_inds)}
    test_db_read_inds = [db_read_inds_map[x] for x in test_ids]

    song_meta_test = song_meta[song_meta["id"].isin(test_ids)]
    test_subgenre_tags = song_meta_test.set_index("id")[
        "song_gn_dtl_gnr_basket"
    ].to_dict()

    # Calculate top-50 sub-genre tags in test set songs
    subgenre_co = {}
    for song, tags in test_subgenre_tags.items():
        for tag in tags:
            if tag in subgenre_co:
                subgenre_co[tag].append(db_read_inds_map[song])
            else:
                subgenre_co[tag] = [db_read_inds_map[song]]
    ranked_sg = sorted([(len(v), k) for k, v in subgenre_co.items()])[-50:]
    top_50_sg = set([p[1] for p in ranked_sg])
    test_subggenre_top_50 = {
        db_read_inds_map[k]: [x for x in v if x in top_50_sg]
        for k, v in test_subgenre_tags.items()
    }

    # Split test songs further into train/validation/test for sub-genre task
    mlb = MultiLabelBinarizer()
    binarized_gnr = mlb.fit_transform(list(test_subggenre_top_50.values()))
    melon_annotations = pd.DataFrame(
        binarized_gnr, index=list(test_subggenre_top_50), columns=mlb.classes_
    )

    msss = MultilabelStratifiedShuffleSplit(test_size=0.2, random_state=0, n_splits=1)
    r = list(msss.split(test_db_read_inds, binarized_gnr))
    sg_train_ids_tmp = [test_db_read_inds[i] for i in r[0][0]]
    sg_test_ids = [test_db_read_inds[i] for i in r[0][1]]
    msss = MultilabelStratifiedShuffleSplit(test_size=0.25, random_state=0, n_splits=1)
    r = list(msss.split(sg_train_ids_tmp, binarized_gnr[r[0][0]]))
    sg_train_ids = [test_db_read_inds[i] for i in r[0][0]]
    sg_val_ids = [test_db_read_inds[i] for i in r[0][1]]
    sg_split = {"train": sg_train_ids, "val": sg_val_ids, "test": sg_test_ids}
    return melon_annotations, sg_split


def get_genre_co(sound_tags):
    """
    Creates dictionary of genre co-occurrences
    """
    genre_co = {}
    for song, tags in sound_tags.items():
        for tag in tags:
            if tag in genre_co:
                genre_co[tag].append(song)
            else:
                genre_co[tag] = [song]
    return genre_co


def get_artist_co(song_meta):
    """
    Creates dictionary of artist co-occurrences
    """
    artist_co = {}
    for _, row in song_meta.iterrows():
        for artist in row["artist_id_basket"]:
            if artist in artist_co:
                artist_co[artist].append(row["id"])
            else:
                artist_co[artist] = [row["id"]]
    return artist_co


def create_data():
    """
    Main function for creating splits and CF vectors, writing them out to the data
    folder.
    """
    # Load Melon data
    db_inds = pickle.load(open("%s/cache_inds.pkl"%CACHE_DIR, "rb"))
    db_inds_set = set(db_inds)
    song_meta = pd.read_json("%s/song_meta.json" % META_DIR)
    song_meta = song_meta[song_meta["id"].isin(db_inds_set)]
    genre_w2v_map = json.load(open(GENRE_MAP_FILE, encoding="utf-8"))
    playlists = pd.concat(
        [
            pd.read_json("%s/%s.json" % (META_DIR, split))
            for split in ["train", "val", "test"]
        ]
    )

    # Get genre tags
    sound_tags = get_sound_tags(song_meta, genre_w2v_map)
    pickle.dump(sound_tags, open("%ssound_tags.pkl" % SAVE_DATASET_LOCATION, "wb"))

    # Run Melon split
    split_ids, cf_vecs = get_data_split(sound_tags)
    pickle.dump(split_ids, open("%ssplit_ids.pkl" % SAVE_DATASET_LOCATION, "wb"))
    pickle.dump(cf_vecs, open("%scf_vecs.pkl" % SAVE_DATASET_LOCATION, "wb"))

    _, val_ids, test_ids = split_ids

    # Get Melon-50 data
    melon_annotations, sg_split = get_melon_sg_split(test_ids, song_meta, db_inds)
    pickle.dump(
        (melon_annotations, sg_split),
        open("%smelon_sg_data.pkl" % SAVE_DATASET_LOCATION, "wb"),
    )

    # Calculate validation pairs for A-IM and Hybrid models
    val_pairs = generate_pairs(
        val_ids, list(playlists["songs"].values))
    pickle.dump(val_pairs, open("%sval_pairs.pkl" % SAVE_DATASET_LOCATION, "wb"))

    genre_co = get_genre_co(sound_tags)
    val_genre_pairs = generate_pairs(val_ids, list(genre_co.values()))
    pickle.dump(
        (genre_co, val_genre_pairs),
        open("%sgenre_co_data.pkl" % SAVE_DATASET_LOCATION, "wb"),
    )

    artist_co = get_artist_co(song_meta)
    val_artist_pairs = generate_pairs(
        val_ids, list(artist_co.values()), exclude=False
    )
    pickle.dump(
        (artist_co, val_artist_pairs),
        open("%sartist_co_data.pkl" % SAVE_DATASET_LOCATION, "wb"),
    )


if __name__ == "__main__":
    create_data()
