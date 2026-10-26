import pytest

from conftest import DATA

from rageval.judges.cache import JudgeCache
from rageval.loader import DatasetError, load_dataset


def test_loads_sample_dataset():
    samples = load_dataset(DATA / "sample_qa.jsonl")
    assert len(samples) == 12
    assert samples[0].relevant_ids == ["pricing.md#2"]
    assert samples[0].answer is None


def test_csv_with_json_list_columns(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text('id,question,relevant_ids,answer\n'
                    'q1,What is X?,"[""doc#1"", ""doc#2""]",X is Y.\n', encoding="utf-8")
    [sample] = load_dataset(path)
    assert sample.relevant_ids == ["doc#1", "doc#2"]
    assert sample.answer == "X is Y."


@pytest.mark.parametrize("content, message", [
    ('{"id": "a", "question": "q"}\n{"id": "a", "question": "q2"}\n', "duplicate id 'a'"),
    ('{"id": "a", "question": "q"}\nnot json\n', "line 2: invalid JSON"),
    ('{"id": "a", "question": "   "}\n', "record 1"),
    ("\n\n", "contains no samples"),
])
def test_invalid_jsonl(tmp_path, content, message):
    path = tmp_path / "d.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_unsupported_format(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(DatasetError, match="unsupported"):
        load_dataset(path)


def test_cache_roundtrip_and_key_isolation(tmp_path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    key_a = JudgeCache.make_key("openai", "model-a", "sys", "user", {"s": 1})
    key_b = JudgeCache.make_key("openai", "model-b", "sys", "user", {"s": 1})
    assert key_a != key_b
    cache.set(key_a, {"claims": ["x"]})
    assert cache.get(key_a) == {"claims": ["x"]}
    assert cache.get(key_b) is None
    cache.close()
    # Persists across connections.
    reopened = JudgeCache(tmp_path / "c.sqlite")
    assert reopened.get(key_a) == {"claims": ["x"]}
    reopened.close()
