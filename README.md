# Evaluating Contrastive Methodologies for Music Representation Learning Using Playlist Data

This repository contains code for reproducing the results of the above submission to ICASSP. In this work, we use contrastive training regimes to learn representations of musical audio, with supervision signal provided by different components of the [Melon Playlist Dataset](https://mtg.github.io/melon-playlist-dataset). 

## Environment setup
The necessary packages for running this code can be found in `requirements.txt`. We recommend setting up a new environment (of Python version 3.6 or earlier) with these packages installed: we use Lightning Memory-Mapped Databases (LMDBs) for efficient data loading, and the ml_pyxis package for using LMDBs in Python requires a Python version of 3.6 or earlier.

## Data pre-processing
After downloading the Melon Playlist Dataset, there are two scripts in the `preprocessing` folder that need to be run to prepare for pre-training. The first is `melon_LMDB_creation.py`, which processes the Melon spectrogram and stores them in a LMDB cache. The second is `data_split.py`, which splits the data into train/validation/test datasets and calculates the collaborative filtering data used for mode <em>P</em> in the paper. The directory containing the Melon data must be updated in each script. Before running these scripts (and all others in the repo) it is necessary to cd into the PlaylistContrastive directory.

As noted in the paper, the upload of the Melon Playlist Dataset used in this work had data errors which meant that approximately 1% of the songs in the dataset were not used. More specifically, several of the .npy files containing spectrograms were either missing or corrupted. These files were 225.npy, 227.npy, 229.npy, 230.npy, 276.npy, and 277.npy. The code used to exclude the relevant songs can be found here [insert link]. 

## Running pre-training
Contrastive pre-training is run by the command `python train.py --config config.yml`. All different parameter settings described in the paper can be produced by varying the configuration in the `config.yml` file. There are three key parameters which control the contrastive training strategy:

### Cross-modal parameters
The first is `contrast_combos`, which is used for the methods involving cross-modal contrast, i.e. **CM** and **Hybrid**, as well as for the self-supervised approach _SS_. This should be a list of lists containing two data modes, with the eligible pairs as follows: 

  `[audio, audio_self]`, `[audio, cf]`, `[audio, genre_w2v]`, and `[cf, genre_w2v]`
  
`cf` corresponds to mode _P_, `genre_w2v` corresponds to _G_, and `audio_self` is used for the self-supervised configuration _SS_. For example, for the _A,P_ strategy, we would use `contrast_combos: [[audio, cf]]`, while for _A,G,P_ we would have `contrast_combos: [[audio, cf], [audio, genre_w2v], [cf, genre_w2v]]`. For no cross-modal comparison (i.e. **A-IM** methods) leave `contrast_combos: []`, and for _SS_ we use `contrast_combos: [audio, audio_self]`.

### Audio intra-modal parameters
The other key parameter is `audio_contrast_params[pair_generation_method]`, i.e. the method for intra-modal audio comparisons, which is used in the **A-IM** and **Hybrid** strategies. The three possible values of this parameter correspond to the methods described in the paper: `Playlist`, `Artist`, and `Genre`. For no audio comparisons (i.e. the **CM** strategy), leave this parameter blank. 

The mixup parameters can also be found in `audio_contrast_params`. As explained in the paper, `mixup_alpha` is set at 7.0 for **A-IM** and **Hybrid** and 5.0 for **CM**.

### Audio encoder parameter
The architecture of the audio encoder can be set by the `model_params[audio][backbone_type]` parameter, as either `resnet` or `sc_cnn`.

### Other parameters
The `wandb_params` are used for tracking model training behaviour in Weights & Biases, and the `run_name` is also used for saving model checkpoints. To train on multiple GPUs, update the `devices` setting. Most other parameters control hyperparmeters such as the number of epochs, batch size, learning rate, and model architectures. Also included are the directories of the data files: the only one of these which needs to be updated is `dataloader_params[audio_pair_files][playlist_file_dir]`, which should point to the `kakao_meta` directory in the Melon data. 

## Running downstream tasks
To be added in due course.
