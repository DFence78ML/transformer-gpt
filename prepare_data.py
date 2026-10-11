from pathlib import Path
import random
import re
import json
import hashlib
import itertools
from itertools import islice
import time
import numpy as np
from collections import defaultdict

SEED = 42

RAW_FILE =[
    #Path("data/raw/books/tiny_input.txt"),
    #Path("data/raw/books/duplicate_test.txt"),
    Path("data/raw/books/minhash_benchmark.txt")
]

OUTPUT_DIR = Path(
    "data/processed"
)

TRAIN_FILE = (
    OUTPUT_DIR / "train.txt"
)

VAL_FILE = (
    OUTPUT_DIR / "val.txt"
)

STATS_FILE = Path(
    "data/stats.json"
)

DOCUMENTS_FILE = Path(
    "data/documents.jsonl"
)

DOCUMENTS_IDX = Path(
    "data/documents_idx.jsonl"
)

VAL_RATIO = 0.05
MIN_CHARS = 100
SEPARATOR = "<|endoftext|>"

K = 3
HASH_FUNC_NUM = 32
BANDS = 8
ROWS = 4
assert int(HASH_FUNC_NUM / (BANDS * ROWS)) == 1

SIM_THRESH = 0.8

MAX_STORIES = None

p = 2147483647

def create_shingles(story,k):
    shingles = set()
    story = story.lower().strip().split()
    if len(story) >= k:
        for i in range(len(story) - (k-1)):
            pointer = i + k
            split = " ".join(story[i:pointer])
            shingles.add(split)
    return shingles

def jaccard_similarity(shingles_a, shingles_b):
    intersection = shingles_a & shingles_b 
    union = shingles_b | shingles_a
    if len(union) == 0:
        return 0
    else:
        jaccard = len(intersection) / len(union) 
        return jaccard

def hash_shingle(shingle):
    encoded_shingle = str(shingle).encode(encoding = "utf-8")
    shingle_hash = hashlib.sha256(encoded_shingle).digest()
    hash_num = int.from_bytes(shingle_hash, byteorder="little")
    return hash_num % p

def create_hash_params(num_hashes, seed):
    rng = random.Random(seed)
    param = []
    for _ in range(num_hashes):
        a = rng.randrange(1,p)
        b = rng.randrange(0,p)
        param.append([a,b])
    return param, p 

def create_minhash_signature(shingles, hash_params, p):
    result = []
    hashed_shingles = []
    assert len(shingles) > 0
    for shingle in shingles:
        assert shingle != ""
        hashed_shingles.append(hash_shingle(shingle))
    hashed_shingles = np.array(hashed_shingles, dtype = "uint64")
    hash_params = np.array(hash_params)
    a = hash_params.T[0]
    b = hash_params.T[1]
    result = np.mod(a[:, None] * hashed_shingles + b[:, None], p) 
    min_hash = result.min(axis = 1)
    return min_hash.tolist()

def compare_hashes(signature_a, signature_b):
    same = 0
    assert len(signature_a) == len(signature_b)
    for i in range(len(signature_a)):
        if signature_a[i] == signature_b[i]:
            same+=1
    estimate = same / len(signature_a)
    return estimate

def create_lsh_buckets(buckets, signature, doc_id, bands, rows):
    if len(buckets) == 0:
        for i in range(bands):
            buckets[i] = {}
    for i in range(bands):
        if tuple(signature[i*rows:i*rows + rows]) in buckets[i].keys():
            old = buckets[i][tuple(signature[i*rows:i*rows + rows])]
            old.append(doc_id)
            buckets[i][tuple(signature[i*rows:i*rows + rows])] = old
        else:
            buckets[i][tuple(signature[i*rows:i*rows + rows])] = [doc_id]
def extract_pairs(buckets):
    candidate_pairs = set()
    for bands in buckets:
        for key in buckets[bands]:
            if len(buckets[bands][key]) > 1:
                out = list(itertools.combinations(buckets[bands][key],2))
                for i in range(len(out)):
                    candidate_pairs.add(out[i])
    return candidate_pairs

def check_candidates(can_pairs, documents_read, idxs, near_dupes):
        pairs = list(can_pairs)
        shingle_nums = set()
        shingles_cache = {}
        for pair in pairs:
            doc_a, doc_b = pair
            if doc_a not in shingle_nums:
                shingle_nums.add(doc_a)
                story_a = get_document(doc_a, idxs, documents_read)
                shingles_a = create_shingles(story_a, K)
                shingles_cache[doc_a] = shingles_a
            if doc_b not in shingle_nums:
                shingle_nums.add(doc_b)
                story_b = get_document(doc_b, idxs, documents_read)
                shingles_b = create_shingles(story_b, K)
                shingles_cache[doc_b] = shingles_b
            
        for pair in pairs:
            shingles_a = shingles_cache[pair[0]]
            shingles_b = shingles_cache[pair[1]]
            sim = jaccard_similarity(shingles_a, shingles_b)
            if sim >= SIM_THRESH:
                near_dupes.append(pair)
        return near_dupes

def write_documents(file, story, doc_id):
        file.write(
            json.dumps(
                {"id": doc_id,
                "text": story}
            ) + "\n"
        )

def get_document(pos, idxs, documents):
        byte_pos = idxs[pos]
        documents.seek(byte_pos)
        line = documents.readline()
        assert line != ""
        line = json.loads(line)
        return line["text"]

def find(x,parent):
     if parent[x] == x:
        return x
     else:
        parent[x] = find(parent[x],parent)
        return parent[x]

def union(a,b, parent, size):
     root_a = find(a, parent)
     root_b = find(b, parent)

     if size[root_a] >= size[root_b]:
        parent[root_b] = root_a
        size[root_a] += size[root_b]
        #del size[root_b]
     else:
        parent[root_a] = root_b
        size[root_b] += size[root_a]

def build_groups(parent):
    groups = defaultdict(list)
    for doc_id, roots in enumerate(parent):
        root = find(doc_id,parent)
        groups[root].append(doc_id)
    return groups

def survivors(groups):
    keep_ids = set()
    removed_ids = set()
    #print(groups)
    for x, key in enumerate(groups):
        ids = list(groups[key])
        keep_ids.add(min(ids))
        ids.remove(min(ids))
        for i in range(len(ids)):
            removed_ids.add(ids[i])
    return keep_ids, removed_ids

def exact_near_duplicates(shingles):
    near_dupes = set()
    for i in range(len(shingles)):
        compare = 0
        while compare < len(shingles)-1:
            if i != compare:
                sim = jaccard_similarity(shingles[i], shingles[compare])
                if sim >= SIM_THRESH:
                    near_dupes.add(tuple(sorted([compare,i])))
            compare+=1
    return near_dupes

def deduplicate(story, seen_hashes):
        encoded_story = story.encode(encoding = "utf-8")
        story_hash = hashlib.sha256(encoded_story).digest()
        if story_hash not in seen_hashes:
            seen_hashes.add(story_hash)
            return story
        else:
            return None 

def emptycheck(story):
    if story == "":
        return True
    else:
        return False

def clean_text(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.replace(
        "\x00",
        ""
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()

def read_stories(issues):
    seen_hashes = set()
    for raw_file in RAW_FILE:
        with open(
            raw_file,
            "r",
            encoding="utf-8"
        ) as f:

            buffer = ""

            for line in f:

                buffer += line
                while SEPARATOR in buffer:

                    story, buffer = (
                        buffer.split(
                            SEPARATOR,
                            1
                        )
                    )

                    story = clean_text(
                        story
                    )

                    issues["raw"] += 1

                    empty = emptycheck(story)
                    if empty:
                        issues["empty"] += 1
                    else:
                        story = deduplicate(story, seen_hashes)
                        if story is not None:
                                if len(story) >= MIN_CHARS:
                                    yield story
                                    issues["retained"] += 1
                                else:
                                    issues["short"] +=1
                        elif story is None:
                            issues["duplicate"] += 1

                        

            buffer = clean_text(buffer)

            empty = emptycheck(buffer)

            if empty:
                pass
            else:
                issues["raw"] += 1
                buffer = deduplicate(buffer, seen_hashes)
                if buffer is not None:
                        if len(buffer) >= MIN_CHARS:
                            issues["retained"] += 1
                            yield buffer
                        else:
                            issues["short"]+=1
                elif buffer is None:
                    issues["duplicate"] += 1


def prepare_data():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    #OVERWRITE EACH RUN
    with open(DOCUMENTS_FILE, "w") as f:
        pass
    with open(DOCUMENTS_IDX, "w") as f:
            pass
    random.seed(SEED)

    temp_train = (
        OUTPUT_DIR / "train_temp.txt"
    )

    temp_val = (
        OUTPUT_DIR / "val_temp.txt"
    )

    train_count = 0
    val_count = 0
    issues = {
        "duplicate":0,
        "short":0,
        "retained":0,
        "raw":0,
        "empty":0
    } 

    train_chars = 0
    val_chars = 0

    doc_id = 0

    buckets = {}
    param, p = create_hash_params(HASH_FUNC_NUM,SEED)

    with open(
        DOCUMENTS_IDX,
        "a"
    ) as documents_idx, open(
        DOCUMENTS_FILE,
        "a"
    ) as documents_file:

    #PASS 1
        print("---------------------------")
        print("PASS 1")
        print("---------------------------")
        start_time = time.time()
        size = {}
        parent = {}
        for i, story in enumerate(
            islice(read_stories(issues), MAX_STORIES)
        ):
            position = documents_file.tell()
            write_documents(documents_file, story, doc_id)
            documents_idx.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "position": position
                    }
                ) + "\n"
            )
            parent[doc_id] = doc_id
            size[doc_id] = 1
            shingle = create_shingles(story, K)
            signature = create_minhash_signature(shingle, param, p)
            create_lsh_buckets(buckets, signature, doc_id, BANDS, ROWS)
            doc_id += 1

            if ((i+1) % 10000 ) == 0:
                print(f"Processed {i+1} documents")
                #pass
        assert doc_id > 0
        pairs = extract_pairs(buckets)
        print("Number of documents:", doc_id)
        print("Number of candidate pairs:", len(pairs))
        print("Number of LSH buckets:", sum(len(b) for b in buckets.values()))
        print("Largest bucket:", max(
            len(bucket)
            for band in buckets.values()
            for bucket in band.values()
        ))
        #print(len(pairs))
        end_time = time.time()
        print(end_time - start_time)
    #Pass 2
    print("---------------------------")
    print("PASS 2")
    print("---------------------------")
    with open(
            DOCUMENTS_IDX,
            "r"
        ) as documents_idx_read, open(
            DOCUMENTS_FILE,
            "r"
        ) as documents_read:
            start_time = time.time()
            idxs = {}
            for line_number, line in enumerate(documents_idx_read, start=1):
                convert = json.loads(line)
                idxs[convert["id"]] = convert["position"]
            end_time = time.time()
            print(end_time - start_time)
            start_time = time.time()
            #print(idxs)

            near_dupes = []
            near_dupes = check_candidates(pairs, documents_read, idxs, near_dupes)
            end_time = time.time()
            print(end_time - start_time)
            start_time = time.time()
            #end_time = time.time()
            #print(end_time - start_time)
            #print(near_dupes)
            candidate_pairs = set()
            for i in range(len(near_dupes)):
                candidate_pairs.add(near_dupes[i][0])
                candidate_pairs.add(near_dupes[i][1])
            print(len(candidate_pairs))
            print(len(near_dupes))
            
    #Pass 3
    print("---------------------------")
    print("PASS 3")
    print("---------------------------")
    start_time = time.time()
    for i in range(len(near_dupes)):
        union(near_dupes[i][0], near_dupes[i][1], parent, size)
    #print(parent)
    #print(size)

    groups = build_groups(parent)
    keep_ids, removed_ids = survivors(groups)
    print(len(removed_ids))
    end_time = time.time()
    print(end_time - start_time)
    #print(keep_ids)
    #print(removed_ids)

    #Pass 4
    #write properly
    print("---------------------------")
    print("PASS 4")
    print("---------------------------")
    near_duplicates = 0
    with open(
        DOCUMENTS_FILE,
        "r"
    ) as documents, open(
        temp_train,
        "w",
        encoding="utf-8"
    ) as train_file, open(
        temp_val,
        "w",
        encoding="utf-8"
    ) as val_file:
        start_time = time.time()
        for all, line in enumerate(documents):
            story_info = json.loads(
                line
            )
            if story_info["id"] in removed_ids:
                near_duplicates += 1
            else:
                story = story_info["text"]
                if random.random() > VAL_RATIO:
                    train_file.write(
                        story
                        + "\n"
                        + SEPARATOR
                        + "\n"
                    )

                    train_count += 1
                    train_chars += len(story)

                else:
                    val_file.write(
                        story
                        + "\n"
                        + SEPARATOR
                        + "\n"
                    )

                    val_count += 1
                    val_chars += len(story)
            
                if (train_count + val_count) % 10000 == 0:
                    print(f"Processed {train_count + val_count} files")
    end_time = time.time()
    print(end_time - start_time)
    assert train_count > 0 and val_count > 0
    assert train_chars > 0 and val_chars > 0
    assert issues["retained"] - train_count - val_count - len(removed_ids) == 0
    assert issues["raw"] - train_count - val_count - len(removed_ids) - issues["duplicate"]  - issues["empty"] - issues["short"]== 0
    retained_after_all = train_count + val_count
    train_average_chars_per_doc = train_chars // train_count
    val_average_chars_per_doc = val_chars // val_count

    temp_train.replace(TRAIN_FILE)
    temp_val.replace(VAL_FILE)

    stats = {

        "train_documents":
            train_count,

        "validation_documents":
            val_count,

        "train_characters":
            train_chars,

        "validation_characters":
            val_chars,

        "total_characters":
            train_chars + val_chars,

        "duplicate_docs":
        issues["duplicate"],

        "short_docs":
        issues["short"],

        "raw_docs":
        issues["raw"],

        "val_average_chars_per_doc":
        val_average_chars_per_doc,

        "train_average_chars_per_doc":
        train_average_chars_per_doc,

        "Post de duplication":
        retained_after_all
    }
    STATS_FILE.write_text(
        json.dumps(
            stats,
            indent=4
        ),
        encoding="utf-8"
    )

    print()
    print("Finished!")
    print()
    print(
        f"Train stories: "
        f"{train_count:,}"
    )

    print(
        f"Validation stories: "
        f"{val_count:,}"
    )

    print(
        f"Train characters: "
        f"{train_chars:,}"
    )

    print(
        f"Validation characters: "
        f"{val_chars:,}"
    )

    print(
        f"Duplicate Docs: " 
        f"{issues["duplicate"]:,}"
    )

    print(
        f"Short docs: " 
        f"{issues["short"]:,}"
    )

    print(
        f"Empty docs: " 
        f"{issues["empty"]:,}"
    )

    print(
        f"Raw docs: " 
        f"{issues["raw"]:,}"
    )

    print(
        f"Average characters per train doc: " 
        f"{train_average_chars_per_doc:,}"
    )

    print(
        f"Average characters per val doc: " 
        f"{val_average_chars_per_doc:,}"
    )

    print(
        f"Post de duplication: "
        f"{retained_after_all:,}"
    )

    print(
        f"Near duplicates: "
        f"{len(near_dupes)}"
    )

    print(
        f"Documents removed as near dupes: "
        f"{len(removed_ids)}"
    )

if __name__ == "__main__":
    prepare_data()
    """shingles = []
    issues = {
        "duplicate":0,
        "short":0,
        "retained":0,
        "raw":0,
        "empty":0
    } 
    for raw_file in RAW_FILE:
        with open(
            raw_file,
            "r",
            encoding="utf-8"
        ) as f:
            for i, story in enumerate(
                islice(read_stories(issues), MAX_STORIES)
            ):
                shingles.append(create_shingles(story, K))
                
            exact_close = exact_near_duplicates(shingles)
            print(len(exact_close))"""
    """text = clean_text("The cat sat on the mat")
    text2 = clean_text("The cat sat on the rat")
    text3 = "I love spaceships that fly"
    documents = [text, text2, text3]
    k = 3
    shingles = []
    hash_shingles =[]
    for text in documents:
        shingle = create_shingles(text, k)
        shingles.append(shingle)
        hash_shingles.append(hash_shingle(create_shingles(text, k)))
    param, p = create_hash_params(128,42)
    signatures = []
    for shingle in shingles:
        signatures.append(create_minhash_signature(shingle, param, p))
    print(signatures[0])

    estimate = compare_hashes(signatures[0], signatures[1])
    sim = jaccard_similarity(shingles[0],shingles[1])
    print(estimate)
    print(sim)
    buckets = {}
    near_dupes = []
    bands = 32
    rows = 4
    for i in range(len(signatures)):
        create_lsh_buckets(buckets, signatures[i], i, bands, rows)
    pairs = extract_pairs(buckets)
    dupe_pairs = check_candidates(pairs, documents, k, near_dupes)
    print(dupe_pairs)"""