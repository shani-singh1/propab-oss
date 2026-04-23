from propab.claim_grounding import partition_claims, prose_to_sentences, sentence_grounding_score


def test_sentence_grounding_score_overlap() -> None:
    s = sentence_grounding_score(
        "The model achieved lower validation loss on the benchmark suite.",
        "validation loss benchmark tool_call train_model output_json",
    )
    assert s > 0.2


def test_partition_claims_threshold() -> None:
    g, u = partition_claims(
        [
            "Alpha beta gamma delta epsilon zeta eta theta iota.",
            "Zyxwvutsrqponmlkjihgfedcbaabcdefghijklmnopqrs.",
        ],
        "alpha beta gamma delta epsilon zeta",
        threshold=0.12,
    )
    assert len(g) >= 1
    assert any("Zyx" in x["sentence"] for x in u)


def test_prose_to_sentences() -> None:
    sents = prose_to_sentences(
        {
            "abstract": "First claim here. Second claim there.",
            "introduction": "",
            "discussion": "",
            "conclusion": "",
        }
    )
    assert len(sents) >= 2
