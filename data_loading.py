import pyxis as px
import pickle
import random
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from functools import partial

myrandom = random.Random(22)


def slice_spec(spec, size=256, include_double=False, double_window=625):
    """
    Function for randomly slicing patch(es) from a spectrogram.

    Args:
        spec: Spectrogram to be sliced
        size: Number of timesteps for patch
        include_double: Bool indicating whether to retrieve two patches (for
            self-supervised training)
        double_window: Number of timesteps either side of first patch to retrieve
            second patch

    Returns:
        Either single patch or dictionary containing two patches (if include_double
            is True)
    """
    spec_shape = spec.shape[1]
    max_ind = max(spec_shape - size, size)
    start_ind = random.choice(range(max_ind))
    spec_out = spec[:, start_ind : (start_ind + size)]
    if not include_double:
        return spec_out
    else:
        min_double_start = max(0, start_ind - double_window)
        max_double_start = min(spec_shape - size, start_ind + double_window)
        eligible_double_starts = [
            x
            for x in range(min_double_start, max_double_start)
            if x not in range(start_ind - size, start_ind + size)
        ]
        try:
            double_start_ind = random.choice(eligible_double_starts)
            double_spec_out = spec[:, double_start_ind : (double_start_ind + size)]
        except:
            double_start_ind = random.choice(range(min_double_start, max_double_start))
            double_spec_out = spec[:, double_start_ind : (double_start_ind + size)]
        return {"audio": spec_out, "audio_self": double_spec_out}


def mixup(inp, alpha=7.0, beta=2.0):
    """
    Performs mixup given batch input with mixup gains sampled from beta distribution with
    parameters alpha and beta.
    """
    batch_size = inp.shape[0]
    mixup_gains = torch.from_numpy(np.random.beta(alpha, beta, (batch_size))).reshape(
        (-1, 1, 1)
    )
    mixup_inds = random.sample(range(batch_size), batch_size)
    inp_shuffled = inp[mixup_inds]
    inp_mixed = mixup_gains * inp + (1 - mixup_gains) * inp_shuffled
    return inp_mixed


def tag_processor(tags, max_num_tags=10):
    """
    Utility function for processing genre tags.
    """
    non_neg = [i + 1 for i in tags]
    new_tags = np.zeros(max_num_tags)
    new_tags[: min(len(non_neg), max_num_tags)] = non_neg[:max_num_tags]
    return new_tags


def batch_collate(batch_loaded):
    """
    Utility collator function for batches containing multiple modes of data.
    Transforms dictionary containing data from each batch index to stacked
    tensors for each data mode.
    """
    batch_loaded_dict = {k: [d[k] for d in batch_loaded] for k in batch_loaded[0]}
    out = {
        k: torch.from_numpy(np.stack(v))
        for k, v in batch_loaded_dict.items()
        if k != "genre_w2v"
    }
    if "genre_w2v" in batch_loaded_dict:
        out["genre_w2v"] = torch.tensor(
            np.stack(batch_loaded_dict["genre_w2v"]), dtype=torch.long
        )
    return out


class MelonDataLoader:
    """
    Main data loader class for CM and SS methods.
    """

    def __init__(
        self,
        batch_size,
        lmdb_cache_file,
        lmdb_cache_inds_file,
        data_split_files,
        contrast_bools,
        audio_contrast_params,
        max_num_tags=10,
        double_window=None,
        sound_tags_file=None,
        cf_emb_file=None,
        num_read_ind_batches=600,
        num_workers=6,
    ):
        """
        Args:
            batch_size: Batch size
            lmdb_cache_file: Location of LMDB cache file for Melon data
            lmdb_cache_inds_file: List containing indices of Melon songs in LMDB,
                for mapping to Melon data
            data_split_files: Contains (train,val,test) song indices
            contrast_bools: Indicates which data modes are to be included
            audio_contrast_params: Parameter dictionary indicating whether to apply mixup
            max_num_tags: Number of tags used in genre encoder
            double_window: Number of timesteps either side of first patch to retrieve
                second patch (used in SS)
            sound_tags_file: File containing genre tag data
            cf_emb_file: File containing collaborative filtering song embeddings
            num_read_ind_batches: Number of batches used in data loading
            num_workers: Workers used in data loading
        """

        self.num_workers = num_workers
        if contrast_bools["audio_self"]:  # SS
            self.batch_size = batch_size
        else:
            # Double batch size for CM methods
            self.batch_size = 2 * batch_size

        # Load LMDB
        db_read = px.Reader(lmdb_cache_file, lock=False)
        self.db_read_inds = pickle.load(open(lmdb_cache_inds_file, "rb"))
        db_read_inds_map = {k: i for i, k in enumerate(self.db_read_inds)}
        self.split_ids = pickle.load(open(data_split_files, "rb"))
        self.train_ids, self.val_ids, self.test_ids = self.split_ids

        id_maps = {
            "train": {k: i for i, k in enumerate(self.train_ids)},
            "val": {k: i for i, k in enumerate(self.val_ids)},
        }
        train_set = set(self.train_ids)

        self.audio_contrast_params = audio_contrast_params
        if self.audio_contrast_params["mixup"]:
            self.mixup = partial(
                mixup,
                **{
                    "alpha": self.audio_contrast_params["mixup_alpha"],
                    "beta": self.audio_contrast_params["mixup_beta"],
                }
            )

        self.train_read_inds_batched = np.array_split(
            [x for x in self.db_read_inds if x in train_set], num_read_ind_batches
        )
        self.contrast_bools = contrast_bools
        self.collator = MelonCollator(
            db_read,
            db_read_inds_map,
            id_maps,
            contrast_bools,
            sound_tags_file,
            cf_emb_file,
            max_num_tags,
            double_window,
        )

        self.load_val_data()

    def load_val_data(self):
        """
        Loads validation data for use in all epochs. Loads all validation songs into cache
        and creates random batches using collator function.
        """
        val_set = set(self.val_ids)
        val_sequential_inds = [x for x in self.db_read_inds if x in val_set]
        val_sequential_ind_batches = np.array_split(val_sequential_inds, 20)

        # Data loader for filling val cache
        val_loader = iter(
            DataLoader(
                val_sequential_ind_batches,
                collate_fn=lambda batch_ids: {
                    i: self.collator.load_single(i, split="val") for i in batch_ids[0]
                },
                shuffle=False,
                batch_size=1,
                num_workers=self.num_workers,
            )
        )

        val_time = time.time()
        val_cache = {}
        for dic in val_loader:
            val_cache = {**val_cache, **dic}
        print("Val data loaded %d" % (time.time() - val_time))
        val_ids_shuffled = myrandom.sample(self.val_ids, len(self.val_ids))

        # Data loader for creating validation batches
        val_batch_loader = iter(
            DataLoader(
                val_ids_shuffled,
                collate_fn=lambda batch_ids: self.collator.cached_batch_collate(
                    batch_ids, val_cache
                ),
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=0,
                drop_last=True,
            )
        )

        self.val_batches = [batch_out for batch_out in val_batch_loader]
        val_cache.clear()

        # Apply mixup if needed
        for batch_out in self.val_batches:
            if self.contrast_bools["audio_self"]:
                if self.audio_contrast_params["mixup"]:
                    concat = torch.cat((batch_out["audio"], batch_out["audio_self"]))
                    concat = self.mixup(concat)
                    batch_out["audio"] = concat[: self.batch_size]
                    batch_out["audio_self"] = concat[self.batch_size :]
            else:
                if self.audio_contrast_params["mixup"]:
                    batch_out["audio"] = self.mixup(batch_out["audio"])

    def cache_build_dataloader(self):
        """
        Data loader for building cache at each epoch during training.
        """
        return iter(
            DataLoader(
                self.train_read_inds_batched,
                collate_fn=lambda batch_ids: {
                    i: self.collator.load_single(i, split="train") for i in batch_ids[0]
                },
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=True,
            )
        )

    def remaining_dataloader(self, remaining, cache):
        """
        Data loader for running training on remaining songs after loading is complete.
        """
        return DataLoader(
            list(remaining),
            batch_size=self.batch_size,
            collate_fn=lambda x: (x, self.collator.cached_batch_collate(x, cache)),
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )


class MelonCollator:
    """
    Collator class for data loader, used to process data in each data mode.
    """

    def __init__(
        self,
        db_read,
        db_read_inds_map,
        id_maps,
        contrast_bools,
        sound_tags_file=None,
        cf_emb_file=None,
        max_num_tags=None,
        double_window=None,
    ):
        self.db_read = db_read
        self.db_read_inds_map = db_read_inds_map
        self.max_num_tags = max_num_tags
        self.double_window = double_window

        self.id_maps = id_maps

        self.genre_w2v = contrast_bools["genre_w2v"]
        self.cf = contrast_bools["cf"]
        self.audio_self = contrast_bools["audio_self"]

        # Load relevant data for non-audio modes
        if self.genre_w2v:
            if sound_tags_file is None:
                raise ValueError("Sound tags file must be provided")
            self.sound_tags = pickle.load(open(sound_tags_file, "rb"))

        if self.cf:
            if cf_emb_file is None:
                raise ValueError("CF embedding file must be provided")
            train_cf_embs, val_cf_embs = pickle.load(open(cf_emb_file, "rb"))
            self.cf_embs = {"train": train_cf_embs, "val": val_cf_embs}

    def load_single(self, batch_id, split):
        """
        Loads data for single song based on what modes are needed.
        """
        batch_db_ind = self.db_read_inds_map[batch_id]
        if not self.audio_self:
            spec = slice_spec(self.db_read[batch_db_ind]["spec"])
            out = {"audio": spec}
        # Load two patches for SS
        else:
            out = slice_spec(
                self.db_read[batch_db_ind]["spec"],
                include_double=True,
                double_window=self.double_window,
            )

        # Get G and/or P inputs if necessary.
        if self.cf:
            out["cf"] = self.cf_embs[split][self.id_maps[split][batch_id]]
        if self.genre_w2v:
            out["genre_w2v"] = tag_processor(
                self.sound_tags[batch_id], self.max_num_tags
            )
        return out

    def cached_batch_collate(self, batch_ids, cache):
        """
        Collates batch based on information stored in cache for each song.
        """
        batch_loaded = [cache.pop(i) for i in batch_ids]
        return batch_collate(batch_loaded)


def generate_pairs(train_songs, C, exclude=True):
    """
    Generates pairs at each epoch by randomly iterating through C and
    dividing each list in two.

    Args:
        train_songs: List of eligible songs for inclusion in pairs.
        C: List of lists of songs (e.g. playlists, lists of songs which share an artist)
        exclude: Bool which is set to True when songs should only be allowed to appear
            in one pair.
    """
    pairs = []
    available = set(train_songs).copy()
    song_lists_shuffled = random.sample(C, len(C))
    for song_lst in song_lists_shuffled:
        song_lst_available = [x for x in song_lst if x in available]
        if len(song_lst_available) > 1:
            shuffled = random.sample(
                song_lst_available, 2 * (len(song_lst_available) // 2)
            )
            if exclude:
                for s in shuffled:
                    available.remove(s)
            paired = np.array_split(shuffled, len(shuffled) // 2)
            pairs += [tuple(x) for x in paired]
    return pairs


class MelonDataLoader_AudioPairs:
    """
    Data loader for hybrid and A-IM methods (except SS).
    """

    def __init__(
        self,
        batch_size,
        lmdb_cache_file,
        lmdb_cache_inds_file,
        data_split_files,
        contrast_bools,
        val_pairs_file,
        pair_generation_method,
        playlist_file_dir,
        audio_contrast_params,
        num_read_ind_batches=600,
        num_workers=6,
        sound_tags_file=None,
        max_num_tags=10,
        cf_emb_file=None,
        artist_co_file=None,
        genre_co_file=None,
        **kwargs
    ):
        """
        Args:
            batch_size: Batch size
            lmdb_cache_file: Location of LMDB cache file for Melon data
            lmdb_cache_inds_file: List containing indices of Melon songs in LMDB,
                for mapping to Melon data
            data_split_files: Contains (train,val,test) song indices
            contrast_bools: Indicates which data modes are to be included
            val_pairs_file: Pickle file containing pairs of songs in validation set for
                val evaluation (playlist-based)
            pair_generation_method: Metadata source for generating pairs based on co-occurence
                (i.e. Playlist, Genre, or Artist)
            playlist_file_dir: Directory containing Melon playlist data files
            audio_contrast_params: Parameter dictionary indicating whether to apply mixup
            num_read_ind_batches: Number of batches used in data loading
            num_workers: Workers used in data loading
            sound_tags_file: File containing genre tag data
            max_num_tags: Number of tags used in genre encoder
            cf_emb_file: File containing collaborative filtering song embeddings
            artist_co_file: Tuple containing (artist data, artist val pairs) where artist data
                is the artist equivalent for C
            genre_co_file: Tuple containing (genre data, genre val pairs) where genre data
                is the genre equivalent for C
        """

        self.num_workers = num_workers
        self.batch_size = batch_size

        # Load LMDB
        db_read = px.Reader(lmdb_cache_file, lock=False)
        self.db_read_inds = pickle.load(open(lmdb_cache_inds_file, "rb"))
        db_read_inds_map = {k: i for i, k in enumerate(self.db_read_inds)}
        self.audio_contrast_params = audio_contrast_params
        if self.audio_contrast_params["mixup"]:
            self.mixup = partial(
                mixup,
                **{
                    "alpha": self.audio_contrast_params["mixup_alpha"],
                    "beta": self.audio_contrast_params["mixup_beta"],
                }
            )

        self.split_ids = pickle.load(open(data_split_files, "rb"))
        self.train_ids, self.val_ids, self.test_ids = self.split_ids

        val_set = set(self.val_ids)
        self.pair_generation_method = pair_generation_method
        if pair_generation_method == "Playlist":
            # Load playlist data
            self.C = list(
                pd.concat(
                    [
                        pd.read_json("%s/%s.json" % (playlist_file_dir, split))
                        for split in ["train", "val", "test"]
                    ]
                )["songs"].values
            )
            self.val_pairs = pickle.load(open(val_pairs_file, "rb"))

        # Get artist and genre data if needed
        elif pair_generation_method == "Artist":
            artist_co, self.val_pairs = pickle.load(open(artist_co_file, "rb"))
            self.C = list(artist_co.values())
        elif pair_generation_method == "Genre":
            genre_co, self.val_pairs = pickle.load(open(genre_co_file, "rb"))
            self.C = list(genre_co.values())
            self.song_genres = {}
            for genre, song_set in genre_co.items():
                for song in song_set:
                    if song in self.song_genres:
                        self.song_genres[song].append(genre)
                    else:
                        self.song_genres[song] = [genre]

        self.val_pairs = [
            p for p in self.val_pairs if p[0] in val_set and p[1] in val_set
        ]
        self.val_pairs_set = set(self.val_pairs)
        print("Num val pairs", len(self.val_pairs))

        id_maps = {
            "train": {k: i for i, k in enumerate(self.train_ids)},
            "val": {k: i for i, k in enumerate(self.val_ids)},
        }
        self.train_set = set(self.train_ids)
        self.num_read_ind_batches = num_read_ind_batches
        self.collator = MelonCollator_AudioPairs(
            db_read,
            db_read_inds_map,
            id_maps,
            contrast_bools,
            sound_tags_file,
            cf_emb_file,
            max_num_tags,
        )
        self.load_val_data()

    def load_val_data(self):
        """
        Loads validation data for use in all epochs. Loads all validation songs into cache
        and creates random batches using collator function.
        """

        val_pair_set = set([s for p in self.val_pairs for s in p])
        val_sequential_inds = [x for x in self.db_read_inds if x in val_pair_set]
        val_sequential_ind_batches = np.array_split(val_sequential_inds, 20)

        # Data loader for filling val cache
        val_loader = iter(
            DataLoader(
                val_sequential_ind_batches,
                collate_fn=lambda batch_ids: {
                    i: self.collator.load_single(i, split="val") for i in batch_ids[0]
                },
                shuffle=False,
                batch_size=1,
                num_workers=self.num_workers,
            )
        )

        val_time = time.time()
        val_cache = {}
        for dic in val_loader:
            val_cache = {**val_cache, **dic}
        print("Val data loaded %d" % (time.time() - val_time))
        val_pairs_shuffled = myrandom.sample(self.val_pairs, len(self.val_pairs))

        # Data loader for creating val batches
        val_batch_loader = iter(
            DataLoader(
                val_pairs_shuffled,
                collate_fn=lambda batch_ids: (
                    batch_ids,
                    self.collator.cached_batch_collate(batch_ids, val_cache),
                ),
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=0,
                drop_last=True,
            )
        )

        self.val_genre_masks = []
        self.val_batches = []
        # Apply mixup if necessary
        for batch, batch_out in val_batch_loader:
            audio = batch_out["audio"]
            num_pairs = audio.shape[0] // 2
            anchors = audio[:num_pairs]
            positives = audio[num_pairs:]
            concat = torch.cat((anchors, positives))
            if self.audio_contrast_params["mixup"]:
                concat = self.mixup(concat)

            batch_out["audio"] = concat.to(dtype=torch.float)
            # Initialise genre weight masks for use in semantic weighing
            if self.audio_contrast_params["pair_generation_method"] == "Genre":
                self.val_genre_masks.append(self.get_genre_mask(batch))
            self.val_batches.append(batch_out)
        val_cache.clear()

    def get_genre_mask(self, batch):
        """
        Calculates semantic weighing mask for genre batch.
        """
        batch_flat = [p[0] for p in batch] + [p[1] for p in batch]
        arr = np.stack(
            [np.eye(N=249)[self.song_genres[s]].sum(axis=0) for s in batch_flat]
        )

        # Get shared counts (factor of two will cancel out in scaling so is omitted)
        weights = (arr @ arr.T) * (1 - np.eye(len(batch_flat)))
        batch_counts = arr.sum(axis=1)
        # Get sum of individual counts
        batch_counts_added = np.expand_dims(batch_counts, 0) + np.expand_dims(
            batch_counts, 1
        )
        return torch.from_numpy(weights / batch_counts_added)

    def cache_build_dataloader(self, train_read_inds_batched):
        """
        Data loader for building cache at each epoch during training.
        """
        return iter(
            DataLoader(
                train_read_inds_batched,
                collate_fn=self.collator.cache_build_collate,
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=True,
            )
        )

    def remaining_dataloader(self, remaining, cache):
        """
        Data loader for running training on remaining songs after loading is complete.
        """
        return iter(
            DataLoader(
                list(remaining),
                batch_size=self.batch_size,
                collate_fn=lambda x: (
                    x,
                    self.collator.cached_batch_collate(x, cache),
                ),
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
        )


class MelonCollator_AudioPairs:
    """
    Collator class for data loader, used to process data in each data mode.
    """

    def __init__(
        self,
        db_read,
        db_read_inds_map,
        id_maps,
        contrast_bools,
        sound_tags_file=None,
        cf_emb_file=None,
        max_num_tags=None,
    ):
        self.db_read = db_read
        self.db_read_inds_map = db_read_inds_map

        self.id_maps = id_maps
        self.genre_w2v = contrast_bools["genre_w2v"]
        self.cf = contrast_bools["cf"]
        self.audio_self = False
        self.max_num_tags = max_num_tags

        if self.genre_w2v:
            if sound_tags_file is None:
                raise ValueError("Sound tags file must be provided")
            self.sound_tags = pickle.load(open(sound_tags_file, "rb"))

        if self.cf:
            if cf_emb_file is None:
                raise ValueError("CF embedding file must be provided")
            train_cf_embs, val_cf_embs = pickle.load(open(cf_emb_file, "rb"))
            self.cf_embs = {"train": train_cf_embs, "val": val_cf_embs}

    def load_single(self, batch_id, split):
        """
        Loads data for single song based on what modes are needed.
        """
        batch_db_ind = self.db_read_inds_map[batch_id]
        spec = slice_spec(self.db_read[batch_db_ind]["spec"])
        out = {"audio": spec}
        if self.cf:
            out["cf"] = self.cf_embs[split][self.id_maps[split][batch_id]]
        if self.genre_w2v:
            out["genre_w2v"] = tag_processor(
                self.sound_tags[batch_id], self.max_num_tags
            )
        return out

    def cached_batch_collate(
        self,
        batch_id_pairs,
        cache,
    ):
        """
        Collates batch based on information stored in cache for each song in each pair.
        """

        batch_ids = list(zip(*batch_id_pairs))
        collated = []
        for batch_lst in batch_ids:
            batch_loaded = [cache[i] for i in batch_lst]
            collated.append(batch_collate(batch_loaded))
        out = {}
        for k in collated[0]:
            out[k] = torch.cat([collated[0][k], collated[1][k]])
        return out

    def cache_build_collate(self, batch_ids):
        """
        Collates batch information to add to cache.
        """
        out = {i: self.load_single(i, split="train") for i in batch_ids[0]}
        return out
