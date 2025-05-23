import random, pathlib, tqdm

random.seed(42)
infile = "combined_dataset.jsonl"
out_dir = pathlib.Path("splits")
out_dir.mkdir(exist_ok=True)

train_f = open(out_dir/"train.jsonl", "w")
val_f = open(out_dir/"val.jsonl", "w")

for line in tqdm.tqdm(open(infile, encoding="utf-8")):
    p = random.random()
    if p < 0.80:
        target = train_f
    else:
        target = val_f
    target.write(line)

for f in (train_f, val_f):
    f.close()
