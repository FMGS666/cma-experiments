import yaml

from cma import CMAOptions
from pathlib import Path

def read_cma_settings_from_yaml(
        yaml_file: str | Path
    ) -> tuple[CMAOptions, dict]:
    with open(yaml_file, 'r') as _option_file:
        cma_settings = yaml.safe_load(_option_file)
    budget_multiplier = cma_settings["budget_multipliers"]
    options = cma_settings["options"]
    cma_kwargs = cma_settings["kwargs"] if cma_settings["kwargs"] else dict()
    cma_options = CMAOptions(
        s = options
    )
    return budget_multiplier, cma_options, cma_kwargs