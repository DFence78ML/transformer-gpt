import random
from pathlib import Path

SEED = 42
random.seed(SEED)

OUTPUT_FILE = Path("data/raw/books/minhash_benchmark.txt")

SEPARATOR = "<|endoftext|>"

NUM_UNRELATED = 1000
NUM_NEAR_DUPLICATE_GROUPS = 200
VARIANTS_PER_GROUP = 4
NUM_EXACT_DUPLICATE_PAIRS = 100


# ---------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------

subjects = [
    "the scientist", "the engineer", "the student", "the teacher",
    "the explorer", "the researcher", "the programmer", "the astronaut",
    "the doctor", "the farmer", "the musician", "the pilot",
    "the inventor", "the photographer", "the mathematician"
]

verbs = [
    "examined", "discovered", "studied", "created", "observed",
    "tested", "recorded", "measured", "designed", "explored",
    "calculated", "investigated", "constructed", "analysed"
]

objects = [
    "the strange machine", "the old laboratory", "the distant planet",
    "the experimental device", "the ancient building",
    "the unusual signal", "the complicated equation",
    "the mysterious object", "the new computer", "the large telescope",
    "the underground chamber", "the research station",
    "the mechanical system", "the damaged spacecraft"
]

locations = [
    "in the laboratory", "near the mountain", "inside the observatory",
    "at the research centre", "beside the river",
    "in the university", "on the remote island",
    "inside the workshop", "at the space station",
    "near the old castle", "in the testing facility"
]

adjectives = [
    "carefully", "slowly", "quickly", "quietly", "successfully",
    "unexpectedly", "methodically", "patiently", "accurately"
]

time_phrases = [
    "early in the morning",
    "later that afternoon",
    "during the experiment",
    "after several hours",
    "the following day",
    "during the first test",
    "before sunset",
    "after a short delay"
]


# ---------------------------------------------------------
# Generate a random sentence
# ---------------------------------------------------------

def generate_sentence():
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)
    location = random.choice(locations)
    adjective = random.choice(adjectives)
    time = random.choice(time_phrases)

    return (
        f"{subject} {adjective} {verb} {obj} "
        f"{location} {time}."
    )


# ---------------------------------------------------------
# Generate a completely unrelated document
# ---------------------------------------------------------

def generate_unrelated_story():
    sentences = []

    for _ in range(random.randint(8, 14)):
        sentences.append(generate_sentence())

    return " ".join(sentences)


# ---------------------------------------------------------
# Generate a base document
# ---------------------------------------------------------

def generate_base_story():
    sentences = []

    for _ in range(12):
        sentences.append(generate_sentence())

    return sentences


# ---------------------------------------------------------
# Make a near-duplicate of a document
#
# mutation_rate controls how many sentences are changed.
# ---------------------------------------------------------

def mutate_story(sentences, mutation_rate):
    result = sentences.copy()

    for i in range(len(result)):
        if random.random() < mutation_rate:
            result[i] = generate_sentence()

    return " ".join(result)


# ---------------------------------------------------------
# Generate dataset
# ---------------------------------------------------------

def generate_dataset():

    documents = []

    # ---------------------------------------------
    # 1. Completely unrelated documents
    # ---------------------------------------------

    for _ in range(NUM_UNRELATED):
        documents.append(
            generate_unrelated_story()
        )

    # ---------------------------------------------
    # 2. Near-duplicate groups
    # ---------------------------------------------

    for _ in range(NUM_NEAR_DUPLICATE_GROUPS):

        base = generate_base_story()

        # Original
        documents.append(
            " ".join(base)
        )

        # Very similar
        documents.append(
            mutate_story(base, 0.10)
        )

        # Moderately similar
        documents.append(
            mutate_story(base, 0.20)
        )

        # More heavily modified
        documents.append(
            mutate_story(base, 0.35)
        )

    # ---------------------------------------------
    # 3. Exact duplicate pairs
    #
    # These test your exact SHA-256 deduplication.
    # ---------------------------------------------

    for _ in range(NUM_EXACT_DUPLICATE_PAIRS):

        story = generate_unrelated_story()

        documents.append(story)
        documents.append(story)

    # ---------------------------------------------
    # Shuffle everything
    # ---------------------------------------------

    random.shuffle(documents)

    # ---------------------------------------------
    # Write file
    # ---------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for story in documents:

            f.write(story)
            f.write("\n")
            f.write(SEPARATOR)
            f.write("\n")

    print("Benchmark dataset generated!")
    print()
    print(f"Output: {OUTPUT_FILE}")
    print(f"Documents: {len(documents):,}")
    print()
    print("Expected composition:")
    print(f"  Unrelated:              {NUM_UNRELATED:,}")
    print(
        f"  Near-duplicate groups:  "
        f"{NUM_NEAR_DUPLICATE_GROUPS:,}"
    )
    print(
        f"  Near-duplicate docs:    "
        f"{NUM_NEAR_DUPLICATE_GROUPS * VARIANTS_PER_GROUP:,}"
    )
    print(
        f"  Exact duplicate pairs:  "
        f"{NUM_EXACT_DUPLICATE_PAIRS:,}"
    )
    print(
        f"  Exact duplicate docs:   "
        f"{NUM_EXACT_DUPLICATE_PAIRS * 2:,}"
    )


if __name__ == "__main__":
    generate_dataset()