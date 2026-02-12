import os, json, re, glob
from collections import defaultdict

import torch

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False


def human(n: int) -> str:
    # human-readable with binary units
    units = ["", "Ki", "Mi", "Gi", "Ti"]
    f = float(n)
    for u in units:
        if abs(f) < 1024.0:
            return f"{f:,.2f} {u}B"
        f /= 1024.0
    return f"{f:,.2f} PiB"


def list_safetensors_files(path: str):
    if os.path.isdir(path):
        # HF sharded safetensors: model.safetensors.index.json
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, "r", encoding="utf-8") as f:
                j = json.load(f)
            shard_files = sorted(set(j["weight_map"].values()))
            return [os.path.join(path, s) for s in shard_files]
        # otherwise: all *.safetensors in dir
        return sorted(glob.glob(os.path.join(path, "*.safetensors")))
    else:
        return [path]


def iter_safetensors(path: str):
    # memory-friendly: use get_slice to read shape/dtype without loading full tensor
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            s = f.get_slice(k)
            shape = s.get_shape()
            dtype = s.get_dtype()
            numel = 1
            for d in shape:
                numel *= d
            yield k, numel, str(dtype)


def iter_torch_bin(path: str):
    # NOTE: torch.load will load whole dict into memory.
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(obj)}")

    for k, v in sd.items():
        if torch.is_tensor(v):
            yield k, v.numel(), str(v.dtype)


def group_key(k: str, depth: int = 2) -> str:
    parts = k.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else k


def count_checkpoint(path: str, include: str = None, exclude: str = None, group_depth: int = 2):
    total_numel = 0
    by_dtype = defaultdict(int)
    by_group = defaultdict(int)

    def accept(k: str) -> bool:
        if include and include not in k:
            return False
        if exclude and exclude in k:
            return False
        return True

    # detect format
    if os.path.isdir(path):
        # prefer safetensors if present
        st_files = list_safetensors_files(path)
        if st_files and st_files[0].endswith(".safetensors"):
            if not HAS_SAFETENSORS:
                raise RuntimeError("safetensors not installed. pip install safetensors")
            files = st_files
            for fp in files:
                for k, n, dt in iter_safetensors(fp):
                    if not accept(k): 
                        continue
                    total_numel += n
                    by_dtype[dt] += n
                    by_group[group_key(k, group_depth)] += n
            return total_numel, by_dtype, by_group, files

        # fallback: look for *.bin in dir
        bin_files = sorted(glob.glob(os.path.join(path, "*.bin")))
        if not bin_files:
            raise FileNotFoundError(f"No safetensors or .bin found in dir: {path}")
        # if multiple bin files, just sum them (rare)
        files = bin_files
        for fp in files:
            for k, n, dt in iter_torch_bin(fp):
                if not accept(k):
                    continue
                total_numel += n
                by_dtype[dt] += n
                by_group[group_key(k, group_depth)] += n
        return total_numel, by_dtype, by_group, files

    # file path
    if path.endswith(".safetensors"):
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed. pip install safetensors")
        files = [path]
        for k, n, dt in iter_safetensors(path):
            if not accept(k):
                continue
            total_numel += n
            by_dtype[dt] += n
            by_group[group_key(k, group_depth)] += n
        return total_numel, by_dtype, by_group, files

    # assume torch bin
    files = [path]
    for k, n, dt in iter_torch_bin(path):
        if not accept(k):
            continue
        total_numel += n
        by_dtype[dt] += n
        by_group[group_key(k, group_depth)] += n
    return total_numel, by_dtype, by_group, files


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="checkpoint file or directory")
    ap.add_argument("--include", default=None, help="only count keys containing this substring")
    ap.add_argument("--exclude", default=None, help="exclude keys containing this substring")
    ap.add_argument("--group-depth", type=int, default=2, help="group by key prefix depth, default 2")
    ap.add_argument("--topk", type=int, default=30, help="show top-k groups")
    ap.add_argument("--check-layer-range", default=None, help=r"regex like 'layers\.(2[9-9]|3[0-3])\.' to detect leftovers")
    args = ap.parse_args()

    total, by_dtype, by_group, files = count_checkpoint(
        args.path, include=args.include, exclude=args.exclude, group_depth=args.group_depth
    )

    print(f"Files ({len(files)}):")
    for fp in files:
        try:
            sz = os.path.getsize(fp)
            print(f"  - {fp} ({human(sz)})")
        except:
            print(f"  - {fp}")

    print("\nTotal parameters (numel): {:,}".format(total))
    print("By dtype:")
    for dt, n in sorted(by_dtype.items(), key=lambda x: -x[1]):
        print(f"  {dt:>12s}: {n:,}")

    print(f"\nTop {args.topk} groups (by numel), group_depth={args.group_depth}:")
    for g, n in sorted(by_group.items(), key=lambda x: -x[1])[: args.topk]:
        print(f"  {g:<40s} {n:,}")

    if args.check_layer_range:
        # quick scan: re-run lightweight iterator (safetensors ok; bin loads anyway)
        rx = re.compile(args.check_layer_range)
        hits = 0
        if os.path.isdir(args.path):
            st_files = list_safetensors_files(args.path)
            if st_files and st_files[0].endswith(".safetensors"):
                for fp in st_files:
                    with safe_open(fp, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            if rx.search(k):
                                hits += 1
                print(f"\nRegex leftover scan hits: {hits} keys matched {args.check_layer_range}")
            else:
                # bin dir: load via count_checkpoint already; keep it simple
                print("\nRegex scan for .bin directories not implemented separately (use file path).")
        else:
            # single file: bin or safetensors
            if args.path.endswith(".safetensors"):
                with safe_open(args.path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if rx.search(k):
                            hits += 1
            else:
                obj = torch.load(args.path, map_location="cpu")
                sd = obj.get("state_dict", obj.get("model", obj)) if isinstance(obj, dict) else {}
                for k in sd.keys():
                    if rx.search(k):
                        hits += 1
            print(f"\nRegex leftover scan hits: {hits} keys matched {args.check_layer_range}")


if __name__ == "__main__":
    main()

