import pathlib
from omegaconf import OmegaConf


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)

stageone_default_config = load_config("stageone_default_config")
stagetwo_default_config = load_config("stagetwo_default_config")
pairvpr_speed = load_config("pairvpr_speed")

def load_and_merge_config(config_name: str):
    default_config = OmegaConf.create(stageone_default_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)