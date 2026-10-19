import pandas as pd

from utils import *
from expconf import ExpConfig
args = ExpConfig.from_yaml()

from selector import Selector

@CustomArgs.wrapper(
    JOB_NAME=f"select_data_{args.exp}",
    WORLD_SIZE=1,
    PARAMS={},
    onlocal=args.select_onlocal,
)
def data_select():
    selector = Selector()

    for sconf in args.select_configs:
        dataset = selector.select(sconf.strategy, sconf.budget, **sconf.kwargs)
        save_file_data(dataset, sconf.data_path)

if __name__ == "__main__":
    data_select()