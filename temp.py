import json, glob

files = sorted(glob.glob('/data/home/cuong/SLM_REASONING/data/math12K_merged_loglikelihood_part*.json'))
for path in files:
    with open(path) as f:
        data = json.load(f)
    removed = 0
    for record in data:
        for answer in record.get('answers', []):
            llh_keys = [k for k in list(answer.keys()) if k.endswith('entropy')]
            for k in llh_keys:
                del answer[k]
                removed += 1
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'{path}: removed {removed} _llh fields')