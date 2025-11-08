import os
from omegaconf import OmegaConf
from typing import List

def load_configs(cfgs: List[str] = None):
    """
    Merge a list of YAML files into a single OmegaConf object.

    Args:
        yaml_files (List[str]): List of paths to YAML files.

    Returns:
        OmegaConf: Merged configuration object.
    """ 
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

    if cfgs is None:
        cfgs = [
            os.path.join(repo_root, 'configs/path.yaml'),
            os.path.join(repo_root, 'configs/model.yaml')
        ]
    
    merged_conf = OmegaConf.create()  # start with empty config
    for cfg in cfgs:
        conf = OmegaConf.load(cfg)
        merged_conf = OmegaConf.merge(merged_conf, conf)
    
    return merged_conf
