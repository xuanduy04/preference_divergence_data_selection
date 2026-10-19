from pathlib import Path

from utils import *
from expconf import *
args = ExpConfig.from_yaml("expconf/config.yaml")

from selector import Selector
args.all_data()
selector = Selector(normlize=False)
df = selector.df

df["DMPO_PDT"] = df["PDT"] * 3

dmpo_data = df[["instruction", "chosen", "rejected", "DMPO_PDT"]].reset_index("dataid").to_dict(orient="records")

save_path = args.data_dir / "dmpo.jsonl"

save_file_data(dmpo_data, save_path)

print(df["DMPO_PDT"].describe())

