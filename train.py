"""
Main script for running contrastive pre-training.
"""
import yaml
import sys

sys.path.append(".")
from trainers import Trainer, Trainer_AudioPairs
import argparse
from pathlib import Path


def main(config):
    # Initialise appropriate trainer based on config
    if config["audio_contrast_params"]["pair_generation_method"] is not None:
        trainer = Trainer_AudioPairs(**config)
    else:
        trainer = Trainer(**config)

    model_checkpoints_folder = Path(trainer.save_model_loc, trainer.run_name)
    if not model_checkpoints_folder.exists():
        model_checkpoints_folder.mkdir()

    # Save config in run folder for future reference
    with open("%s/config.yaml" % model_checkpoints_folder, "w") as f:
        yaml.dump(config, f)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append")

    config_file = parser.parse_args().config

    with open(config_file[0], "r") as in_f:
        config = yaml.safe_load(in_f)
    main(config)
