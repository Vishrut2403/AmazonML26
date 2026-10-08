"""Shared helpers: TSV loading and the challenge's macro F0.5 metric."""
import polars as pl


def read_tsv(path):
    """Read a challenge TSV as all-string columns. quote_char=None because names/addresses contain stray quotes."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def f05(pred, truth):
    """F0.5 for one Source 1 entity. pred/truth are sets of matched S2/S3 ids.

    Empty truth: 1.0 only if pred is also empty (singleton rule). Empty pred with non-empty truth: 0.0.
    """
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, truth):
    """Mean F0.5 over every Source 1 id in truth. pred/truth: dict s1_id -> set of ids; missing pred = empty."""
    return sum(f05(pred.get(k, set()), v) for k, v in truth.items()) / len(truth)


if __name__ == "__main__":
    # the worked example from the problem statement
    assert round(f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}), 3) == 0.714
    assert f05(set(), set()) == 1.0 and f05({"S2-1"}, set()) == 0.0 and f05(set(), {"S2-1"}) == 0.0
    assert macro_f05({"a": set()}, {"a": set(), "b": {"S2-1"}}) == 0.5
    print("ok")
