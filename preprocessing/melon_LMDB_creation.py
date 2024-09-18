"""
This script transforms the Melon spectrograms to magnitudes and writes
them to a LMDB file.
"""

import pyxis as px
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

# Directory containing extracted Melon data
MELON_DIR = "..."

# Directory for saving the LMDB file
SAVE_DIR = "./data/Melon"


def get_file(song_id):
    if len(str(song_id)) <= 3:
        return "%s/arena_mel/0/%d.npy" % (MELON_DIR, song_id)
    else:
        return "%s/arena_mel/%s/%s.npy" % (MELON_DIR, str(song_id)[:-3], song_id)


test = pd.read_json("%s/kakao_meta/test.json" % MELON_DIR)
val = pd.read_json("%s/kakao_meta/val.json" % MELON_DIR)
train = pd.read_json("%s/kakao_meta/train.json" % MELON_DIR)
playlists = pd.concat([train, val, test])

all_songs = set([s for l in playlists["songs"] for s in l])

# Exclude songs with missing/corrupted spectrogram files
excl = (
    list(range(225000, 226000))
    + list(range(227000, 228000))
    + list(range(276000, 277000))
    + list(range(229000, 231000))
)
db_inds = sorted(all_songs.difference(excl))

transform = lambda spec: np.expand_dims(
    np.log10(0.01 + librosa.db_to_amplitude(spec)), axis=0
)
lmdb_dl = iter(
    DataLoader(
        db_inds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda song_ids: {
            "spec": transform(np.load(get_file(song_id))) for song_id in song_ids
        },
        num_workers=12,
        prefetch_factor=2,
        drop_last=False,
    )
)

with px.Writer(
    dirpath="%s/cache" % SAVE_DIR, map_size_limit=260000, ram_gb_limit=50
) as db:
    for i, out in enumerate(tqdm(lmdb_dl)):
        db.put_samples(out)

# Save cache indices
with open("%s/cache_inds.pkl" % SAVE_DIR, "wb") as f:
    pickle.dump(db_inds, f)
