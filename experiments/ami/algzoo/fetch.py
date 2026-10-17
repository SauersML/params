"""Download every AlgZoo checkpoint (fully trained, all seeds) and log file from ARC's public bucket."""
import json
import os
import sys
import urllib.parse
import urllib.request

BUCKET = "https://storage.googleapis.com/storage/v1/b/arc-ml-public/o"
OUT = os.path.expanduser("~/mpd-data/ami/algzoo/weights")


def list_objects(prefix):
    names, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q)) as r:
            page = json.load(r)
        names += [(it["name"], int(it["size"])) for it in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def main():
    objs = list_objects("alg/")
    with open(os.path.join(OUT, "listing.json"), "w") as f:
        json.dump(objs, f)
    full = [n for n, _ in objs if n.startswith("alg/zoo/") and n.endswith(f"_{2**30}_" + n.split("_")[-1])]
    full += ["alg/one_layer_16_hidden_94_acc_2nd_argmax.pth"]
    print(len(objs), "objects;", len(full), "fully trained checkpoints", file=sys.stderr)
    for n in full:
        dst = os.path.join(OUT, os.path.basename(n))
        if os.path.exists(dst):
            continue
        urllib.request.urlretrieve("https://storage.googleapis.com/arc-ml-public/" + n, dst)


if __name__ == "__main__":
    main()
