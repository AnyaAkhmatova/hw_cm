from enum import Enum

from .wandb import WanDBWriter


def get_visualizer(config, logger):
    return WanDBWriter(config, logger)

