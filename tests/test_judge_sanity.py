"""The faithfulness metric must tell faithful answers from hallucinated ones."""

from conftest import DATA

from rageval.evaluator import Evaluator
from rageval.loader import load_dataset


async def test_faithful_beats_hallucinated_on_sample_data(config, judge):
    samples = {s.id: s for s in load_dataset(DATA / "sample_with_answers.jsonl")}
    run = await Evaluator(config, judge).evaluate(list(samples.values()))
    scores = {s.sample_id: s.metrics["faithfulness"].score for s in run.samples}

    # a01 restates the context; a02 invents a price, seat count and feature.
    assert scores["a01"] == 1.0
    assert scores["a02"] == 0.0
    # a04 is fully grounded in its single chunk.
    assert scores["a04"] == 1.0


async def test_bad_retrieval_shows_up_in_context_recall(config, judge):
    samples = {s.id: s for s in load_dataset(DATA / "sample_with_answers.jsonl")}
    run = await Evaluator(config, judge).evaluate([samples["a03"]])
    metrics = run.samples[0].metrics
    # The retention chunk was never retrieved, so the ground truth isn't covered.
    assert metrics["context_recall"].score == 0.0
    assert metrics["recall@5"].score == 0.0
