import gzip, json

with gzip.open("D:\\github\\git repositories\\new_Updated_project\\ml\\data\\reviews\\Amazon_Fashion.jsonl.gz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(json.loads(line))
        break