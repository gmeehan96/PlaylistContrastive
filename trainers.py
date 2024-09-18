"""
This script defines the trainer classes used in contrastive pre-training.
"""

import sys

sys.path.append(".")
from data_loading import *
import random
import torch
import numpy as np
import wandb
import time
import gc
import loss_utils
from model_utils import model_functions
from functools import partial
from itertools import chain
import torch.optim as optim
from pathlib import Path

torch.set_num_threads(6)


class Trainer:
    """
    Trainer class for CM and SS methods.
    """

    def __init__(
        self,
        dataloader_params,
        opt_params,
        wandb_params,
        loss_params,
        model_params,
        save_model_loc,
        contrast_combos,
        audio_contrast_params=None,
        epochs=200,
        early_stop_patience=20,
        min_cache_batches=50,
        devices=[0],
    ):
        """
        Args:
            dataloader_params: Data loader parameters
            opt_params: Optimisation parameters (learning rate and weight decay)
            wandb_params: Parameters used for run tracking in Weights and Biases
            loss_params: Parameters used for defining contrastive loss
            model_params: Parameters for defining models in each data mode
            save_model_loc: Folder for saving model checkpoints and run information
            contrast_combos: Cross-modal combinations for contrastive learning
            audio_contrast_params: Parameters for audio contrast; only used in SS
                scenario
            epochs: Number of training epochs
            early_stop_patience: Patience in early stopping based on validation loss
            min_cache_batches: Number of batches which must be present in cache for
                the model to be trained on a batch (used to ensure diversity in cache)
            devices: List of GPU IDs
        """
        # Process cross-modal contrast combinations
        self.contrast_combos = list(
            set([tuple(sorted(x)) for x in contrast_combos if x[0] != x[1]])
        )
        contrast_tasks = set([task for c in contrast_combos for task in c])
        contrast_bools = {
            task: True if task in contrast_tasks else False for task in model_functions
        }

        dataloader_params = {
            "audio_contrast_params": audio_contrast_params,
            **dataloader_params["shared"],
            **dataloader_params["contrast_files"],
        }

        # Initialise data loader
        self.data_loader = MelonDataLoader(
            contrast_bools=contrast_bools, **dataloader_params
        )

        self.backbone_type = model_params["audio"]["backbone_type"]
        self.wandb_params = wandb_params
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.device = "cuda"
        self.contrast_tasks = [task for task, v in contrast_bools.items() if v]
        if "audio_self" in self.contrast_tasks:
            model_params["audio_self"] = model_params["audio"]
        self.min_cache_batches = min_cache_batches
        self.model_params = model_params
        self.audio_contrast_params = audio_contrast_params

        self.save_model_loc = save_model_loc
        self.run_name = str(wandb_params["run_id"]) + "_" + wandb_params["run_name"]
        self.curr_min_val = 10000
        self.curr_best_epoch = 0

        if self.audio_contrast_params["mixup"]:
            self.mixup = partial(
                mixup,
                **{
                    "alpha": self.audio_contrast_params["mixup_alpha"],
                    "beta": self.audio_contrast_params["mixup_beta"],
                },
            )

        self.models = {}
        self.models = {
            k: torch.nn.DataParallel(
                model_functions[k](model_params[k]).to(self.device),
                device_ids=devices,
            )
            for k in self.contrast_tasks
        }

        self.num_batches_per_epoch = (
            len(self.data_loader.train_ids) // self.data_loader.batch_size
        )

        # Initialise loss function
        self.loss_function = loss_utils.get_loss_function(loss_params)

        # Define L2 weight decay based on backbone type
        opt_params["weight_decay"] = 0 if self.backbone_type == "resnet" else 1e-4

        # Initialise optimizer
        self.optimizer = self.create_optimizer(opt_params)

        self.run = wandb.init(
            project=wandb_params["project"],
            name=self.run_name,
        )

    def create_optimizer(self, opt_params):
        """
        Creates Adam optimizer used in training.
        """
        all_model_params = chain(*[m.parameters() for m in self.models.values()])
        optimizer = optim.Adam(all_model_params, **opt_params)
        return optimizer

    def get_genre_w2v_out(self, tags_input):
        """
        Utility function for genre tag encoder output.
        """
        return self.models["genre_w2v"](tags_input, mask=tags_input.unsqueeze(1))

    def get_audio_out(self, audio_input):
        """
        Utility function for audio encoder output.
        """
        return self.models["audio"](audio_input)[1]

    def calculate_loss(self, batch_out, mode="train"):
        """
        Given a batch, calculate contrastive loss.
        """
        out = {}
        if "genre_w2v" in batch_out:
            out["genre_w2v"] = self.get_genre_w2v_out(
                batch_out["genre_w2v"].to(self.device)
            )[0]

        if "audio_self" in batch_out:
            batch_size = batch_out["audio"].shape[0]
            if mode == "train":
                concat = torch.cat((batch_out["audio"], batch_out["audio_self"]))
                if self.audio_contrast_params["mixup"]:
                    concat = self.mixup(concat)
            else:
                concat = torch.cat((batch_out["audio"], batch_out["audio_self"]))
            out_both = self.get_audio_out(concat.to(self.device, dtype=torch.float))
            out["audio"] = out_both[:batch_size]
            out["audio_self"] = out_both[batch_size:]
        else:
            if self.audio_contrast_params["mixup"] and mode == "train":
                batch_out["audio"] = self.mixup(batch_out["audio"])
            out["audio"] = self.get_audio_out(
                batch_out["audio"].to(self.device, dtype=torch.float)
            )

        out_rest = {
            k: self.models[k](v.to(self.device))
            for k, v in batch_out.items()
            if k not in out
        }
        out = {**out, **out_rest}

        losses = {
            combo: self.loss_function((out[combo[0]], out[combo[1]]))
            for combo in self.contrast_combos
        }

        losses["total"] = torch.sum(torch.stack(tuple(losses.values())))
        return losses

    def loss_wrapper(self, batch_out, mode="train"):
        """
        Wrapper function which calculates loss and updates model weights.
        """
        if mode == "train":
            # Optimize models
            losses = self.calculate_loss(batch_out, mode=mode)
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()
        elif mode == "val":
            with torch.no_grad():
                losses = self.calculate_loss(batch_out, mode=mode)
        torch.cuda.empty_cache()
        return losses

    def train_batch(self, batch, remaining, cache):
        """
        Performs training for a single batch, and updates data cache and list
        of remaining songs.
        """
        batch_out = self.data_loader.collator.cached_batch_collate(batch, cache)
        remaining = remaining.difference(batch)

        losses = self.loss_wrapper(batch_out)
        return losses, remaining

    def train_epoch(self):
        """
        Runs training for a single epoch. To help with data loading efficiency,
        we take advantage of the speed uplift from reading sequential memory, as LMDB
        is optimised for, rather than loading in a random order. We divide the songs
        in the Melon data into sequential blocks of size 1000 and load these blocks in a
        random order into a cache. Once we have a sufficient number of songs loaded for 50
        batches, we start to train the model, removing songs from the cache after they
        have been part of the training batch. We do this at each step in the cache building
        loop, until all songs have been loaded. We then run training on batches created from
        the remaining songs in the cache.
        """
        for m in self.models.values():
            m.train()

        train_losses = []
        remaining = set(self.data_loader.train_ids)
        cache_build_dataloader = self.data_loader.cache_build_dataloader()
        cache = {}
        for i, dic in enumerate(cache_build_dataloader):
            cache = {**cache, **dic}
            del dic
            cached = list(cache)

            if len(cached) > self.min_cache_batches * self.data_loader.batch_size:
                batch = random.sample(cached, self.data_loader.batch_size)
                losses, remaining = self.train_batch(batch, remaining, cache)
                train_losses.append(losses)
                if i % 100 == 0:
                    cache = cache.copy()
                    gc.collect()

        remaining_dataloader = self.data_loader.remaining_dataloader(remaining, cache)
        for i, (batch, batch_out) in enumerate(remaining_dataloader):
            remaining = remaining.difference(batch)
            losses = self.loss_wrapper(batch_out)
            train_losses.append(losses)
            if i % 100 == 0:
                cache = cache.copy()
                gc.collect()

        epoch_losses = loss_utils.get_average_losses(train_losses)
        cache.clear()
        gc.collect()
        return epoch_losses

    def val_epoch(self):
        """
        Calculate validation loss for a single epoch.
        """
        for m in self.models.values():
            m.eval()
        val_losses = []

        for i, val_batch in enumerate(self.data_loader.val_batches):
            val_losses.append(self.loss_wrapper(val_batch, "val"))

        val_losses_out = loss_utils.get_average_losses(val_losses)
        return val_losses_out

    def train(self):
        """
        Main training loop.
        """
        for epoch in range(self.epochs):
            start = time.time()
            train_losses = self.train_epoch()
            val_losses = self.val_epoch()
            logger_dict = {
                "epoch": epoch,
                "train_loss": train_losses["total"],
                "val_loss": val_losses["total"],
            }
            for loss_pair, loss in train_losses.items():
                if loss_pair != "total":
                    logger_dict["train_%s_%s_loss" % loss_pair] = loss
                    logger_dict["val_%s_%s_loss" % loss_pair] = val_losses[loss_pair]
            wandb.log(logger_dict)
            print(
                "Epoch %d Loss: {:5.4f}, Val Loss: {:5.4f}, Time: %d sec".format(
                    train_losses["total"], val_losses["total"]
                )
                % (epoch + 1, time.time() - start)
            )
            if val_losses["total"] < self.curr_min_val:
                self.curr_min_val = val_losses["total"]
                torch.save(
                    self.models["audio"].module.state_dict(),
                    str(
                        Path(
                            self.save_model_loc,
                            self.run_name,
                            f"audio_encoder_epoch_best.pt",
                        )
                    ),
                )
                self.curr_best_epoch = epoch
            if epoch - self.curr_best_epoch >= self.early_stop_patience:
                print("Early stopping threshold reached")
                break

        wandb.finish()


class Trainer_AudioPairs:
    """
    Trainer class for hybrid and A-IM methods (except SS).
    """

    def __init__(
        self,
        dataloader_params,
        opt_params,
        wandb_params,
        loss_params,
        model_params,
        save_model_loc,
        contrast_combos=None,
        audio_contrast_params=None,
        epochs=200,
        early_stop_patience=20,
        min_cache_batches=50,
        devices=[0],
    ):
        """
        Args:
            dataloader_params: Data loader parameters
            opt_params: Optimisation parameters (learning rate and weight decay)
            wandb_params: Parameters used for run tracking in Weights and Biases
            loss_params: Parameters used for defining contrastive loss
            model_params: Parameters for defining models in each data mode
            save_model_loc: Folder for saving model checkpoints and run information
            contrast_combos: Cross-modal combinations for contrastive learning; if None,
                model will be trained using only A-IM method
            audio_contrast_params: Parameters for audio contrast; only used in SS
                scenario
            epochs: Number of training epochs
            early_stop_patience: Patience in early stopping based on validation loss
            min_cache_batches: Number of batches which must be present in cache for
                the model to be trained on a batch (used to ensure diversity in cache)
            devices: List of GPU IDs
        """

        self.backbone_type = model_params["audio"]["backbone_type"]
        self.wandb_params = wandb_params
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.device = "cuda"
        self.min_cache_batches = min_cache_batches
        self.save_model_loc = save_model_loc
        self.run_name = str(wandb_params["run_id"]) + "_" + wandb_params["run_name"]
        self.curr_min_val = 10000
        self.curr_best_epoch = 0
        self.contrast_combos = list(
            set([tuple(sorted(x)) for x in contrast_combos if x[0] != x[1]])
        )
        contrast_tasks = set([task for c in contrast_combos for task in c]).union(
            ["audio"]
        )
        contrast_bools = {
            task: True if task in contrast_tasks else False for task in model_functions
        }
        self.contrast_tasks = [task for task, v in contrast_bools.items() if v]
        self.models = {
            k: torch.nn.DataParallel(
                model_functions[k](model_params[k]).to(self.device),
                device_ids=devices,
            )
            for k in self.contrast_tasks
        }

        self.pair_generation_method = audio_contrast_params["pair_generation_method"]
        dataloader_params = {
            "pair_generation_method": self.pair_generation_method,
            "contrast_bools": contrast_bools,
            "audio_contrast_params": audio_contrast_params,
            **dataloader_params["shared"],
            **dataloader_params["audio_pair_files"],
            **dataloader_params["contrast_files"],
        }
        self.data_loader = MelonDataLoader_AudioPairs(**dataloader_params)

        self.audio_contrast_params = audio_contrast_params

        if self.audio_contrast_params["mixup"]:
            self.mixup = partial(
                mixup,
                **{
                    "alpha": self.audio_contrast_params["mixup_alpha"],
                    "beta": self.audio_contrast_params["mixup_beta"],
                },
            )

        pair_generation_dict = {
            "Playlist": generate_pairs,
            "Artist": partial(generate_pairs, **{"exclude": False}),
            "Genre": generate_pairs,
        }

        self.pair_generator = partial(
            pair_generation_dict[self.pair_generation_method],
            **{
                "C": self.data_loader.C,
                "train_songs": self.data_loader.train_ids,
            },
        )

        self.num_batches_per_epoch = (
            len(self.pair_generator()) // self.data_loader.batch_size
        )

        self.loss_function = loss_utils.get_loss_function(
            loss_params
        )
        opt_params["weight_decay"] = 0 if self.backbone_type == "resnet" else 1e-4
        self.optimizer = self.create_optimizer(opt_params)

        self.run = wandb.init(
            project=wandb_params["project"],
            name=self.run_name,
        )

    def create_optimizer(self, opt_params):
        """
        Creates Adam optimizer used in training.
        """
        all_model_params = chain(*[m.parameters() for m in self.models.values()])
        optimizer = optim.Adam(all_model_params, **opt_params)
        return optimizer

    def get_genre_w2v_out(self, tags_input):
        """
        Utility function for genre tag encoder output.
        """
        return self.models["genre_w2v"](tags_input, mask=tags_input.unsqueeze(1))

    def get_audio_out(self, audio_input):
        """
        Utility function for audio encoder output.
        """
        return self.models["audio"](audio_input)

    def calculate_loss(self, batch_out, mode="train", genre_mask=None):
        """
        Given a batch, calculate contrastive loss.
        """
        out = {}
        weight_mask = genre_mask if self.pair_generation_method == "Genre" else None
        # if len(self.contrast_combos) > 0:
        if "genre_w2v" in batch_out:
            out["genre_w2v"] = self.get_genre_w2v_out(
                batch_out["genre_w2v"].to(self.device)
            )[0]

        num_pairs = batch_out["audio"].shape[0] // 2
        anchors = batch_out["audio"][:num_pairs]
        positives = batch_out["audio"][num_pairs:]
        concat = torch.cat((anchors, positives))
        if self.audio_contrast_params["mixup"] and mode == "train":
            concat = self.mixup(concat)

        out["audio"] = self.get_audio_out(concat.to(self.device, dtype=torch.float))[1]

        out_rest = {
            k: self.models[k](v.to(self.device))
            for k, v in batch_out.items()
            if k not in out
        }
        out = {**out, **out_rest}
        losses = {
            combo: self.loss_function((out[combo[0]], out[combo[1]]))
            for combo in self.contrast_combos
        }

        losses["audio_%s_CO" % self.pair_generation_method] = self.loss_function(
            (out["audio"][:num_pairs], out["audio"][num_pairs:]),
            weight_mask=weight_mask,
        )

        losses["total"] = torch.sum(torch.stack(tuple(losses.values())))
        return losses

    def loss_wrapper(self, batch_out, mode="train", genre_mask=None):
        """
        Wrapper function which calculates loss and updates model weights.
        """

        if mode == "train":
            # Optimize models
            losses = self.calculate_loss(batch_out, mode=mode, genre_mask=genre_mask)
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            losses["total"].backward()
            self.optimizer.step()
        elif mode == "val":
            with torch.no_grad():
                losses = self.calculate_loss(
                    batch_out, mode=mode, genre_mask=genre_mask
                )

        return losses

    def train_batch(self, batch, batch_out, remaining, cache):
        """
        Performs training for a single batch, and updates data cache and list
        of remaining pairs.
        """

        genre_mask = (
            self.data_loader.get_genre_mask(batch)
            if self.pair_generation_method == "Genre"
            else None
        )
        remaining = remaining.difference(batch)
        batch_flat = {s for p in batch for s in p}
        remaining_flat = {s for p in remaining for s in p}
        remove = batch_flat.difference(remaining_flat)
        [cache.pop(s) for s in remove]

        losses = self.loss_wrapper(batch_out, genre_mask=genre_mask)
        return losses, remaining, cache

    def train_epoch(self):
        """
        Runs training for a single epoch. To help with data loading efficiency,
        we take advantage of the speed uplift from reading sequential memory, as LMDB
        is optimised for, rather than loading in a random order. We divide the songs
        in the Melon data into sequential blocks of size 1000 and load these blocks in a
        random order into a cache. Once we have a sufficient number of songs loaded for 50
        batches of pairs, we start to train the model, removing songs from the cache after they
        have been part of the training batch. We do this at each step in the cache building
        loop, until all songs have been loaded. We then run training on batches created from
        the remaining songs in the cache.
        """

        for m in self.models.values():
            m.train()

        train_losses = []
        pairs = self.pair_generator()
        remaining = set(pairs)
        pairs_flat = set([s for p in pairs for s in p])
        train_read_inds_batched = np.array_split(
            [x for x in self.data_loader.db_read_inds if x in pairs_flat],
            self.data_loader.num_read_ind_batches,
        )

        cache_build_dataloader = self.data_loader.cache_build_dataloader(
            train_read_inds_batched
        )
        cache = {}
        for i, dic in enumerate(cache_build_dataloader):
            cache = {**cache, **dic}
            del dic
            cached = [p for p in pairs if len(p) == len([s for s in p if s in cache])]

            if len(cached) > self.min_cache_batches * self.data_loader.batch_size:
                batch = random.sample(cached, self.data_loader.batch_size)
                batch_out = self.data_loader.collator.cached_batch_collate(batch, cache)
                losses, remaining, cache = self.train_batch(
                    batch, batch_out, remaining, cache
                )
                train_losses.append(losses)
                if i % 100 == 0:
                    cache = cache.copy()
                    gc.collect()
        remaining_dataloader = self.data_loader.remaining_dataloader(remaining, cache)
        for i, (batch, batch_out) in enumerate(remaining_dataloader):
            remaining = remaining.difference(batch)
            losses, remaining, cache = self.train_batch(
                batch, batch_out, remaining, cache
            )

            del batch_out
            train_losses.append(losses)
            if i % 100 == 0:
                cache = cache.copy()
                gc.collect()

        epoch_losses = loss_utils.get_average_losses(train_losses)
        cache.clear()
        gc.collect()
        return epoch_losses

    def val_epoch(self):
        """
        Calculate validation loss for a single epoch.
        """

        for m in self.models.values():
            m.eval()
        val_losses = []

        for i, val_batch in enumerate(self.data_loader.val_batches):
            genre_mask = (
                self.data_loader.val_genre_masks[i]
                if self.pair_generation_method == "Genre"
                else None
            )
            val_losses.append(
                self.loss_wrapper(val_batch, "val", genre_mask=genre_mask)
            )

        val_losses_out = loss_utils.get_average_losses(val_losses)
        return val_losses_out

    def train(self):
        """
        Main training loop.
        """
        for epoch in range(self.epochs):
            start = time.time()
            train_losses = self.train_epoch()
            val_losses = self.val_epoch()
            logger_dict = {
                "epoch": epoch,
                "train_loss": train_losses["total"],
                "val_loss": val_losses["total"],
            }
            for loss_pair, loss in train_losses.items():
                if type(loss_pair) == tuple:
                    logger_dict["train_%s_%s_loss" % loss_pair] = loss
                    logger_dict["val_%s_%s_loss" % loss_pair] = val_losses[loss_pair]
                elif loss_pair != "total":
                    logger_dict["train_%s_loss" % loss_pair] = loss
                    logger_dict["val_%s_loss" % loss_pair] = val_losses[loss_pair]

            wandb.log(logger_dict)
            print(
                "Epoch %d Loss: {:5.4f}, Val Loss: {:5.4f}, Time: %d sec".format(
                    train_losses["total"], val_losses["total"]
                )
                % (epoch + 1, time.time() - start)
            )
            if val_losses["total"] < self.curr_min_val:
                self.curr_min_val = val_losses["total"]
                torch.save(
                    self.models["audio"].module.state_dict(),
                    str(
                        Path(
                            self.save_model_loc,
                            self.run_name,
                            f"audio_encoder_epoch_best.pt",
                        )
                    ),
                )
                self.curr_best_epoch = epoch
            if epoch - self.curr_best_epoch >= self.early_stop_patience:
                print("Early stopping threshold reached")
                break
        wandb.finish()
