"""Stage 5: deliberately hidden bugs + verification that PyShrink finds and shrinks them.

Run:  python demo.py      (exit code 0 = all checks passed)
"""
import random
import sys

from pyshrink import (Falsified, RuleBasedStateMachine, dictionaries, given, integers, invariant, lists,
                      minimize, precondition, rule, run_state_machine, text)


# ───────────── Deliberately buggy functions ─────────────

def hidden_bug(xs):
    """Crashes if the length is greater than 3 and it contains a negative even number."""
    if len(xs) > 3 and any(x < 0 and x % 2 == 0 for x in xs):
        raise ValueError("hidden edge case triggered")
    return sum(xs)


def rle_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{j - i}{s[i]}")
        i = j
    return "".join(out)


def rle_decode(s):
    out, num = [], ""
    for ch in s:
        if ch.isdigit():  # BUG: digits in the text itself get mixed up with the counter
            num += ch
        else:
            out.append(ch * int(num))
            num = ""
    return "".join(out)


INT32_MAX = 2**31 - 1


def midpoint_int32(lo, hi):
    total = lo + hi  # BUG: classic binary search overflow; should be lo + (hi - lo) // 2
    if total > INT32_MAX:
        raise OverflowError("int32 overflow: lo + hi")
    return total // 2


def merge_case_insensitive(d):
    out = {}
    for k, v in d.items():
        out[k.lower()] = v  # BUG: overwrites instead of summing values of the same key
    return out


class MinStack:
    """Stack with O(1) get_min."""

    def __init__(self):
        self._items, self._mins = [], []

    def push(self, x):
        self._items.append(x)
        if not self._mins or x < self._mins[-1]:  # BUG: should be <= (repeated minimum)
            self._mins.append(x)

    def pop(self):
        x = self._items.pop()
        if x == self._mins[-1]:
            self._mins.pop()
        return x

    def clear(self):
        self._items.clear()
        self._mins.clear()

    def get_min(self):
        return self._mins[-1] if self._mins else None


# ───────────── Properties ─────────────

@given(lists(integers(-1000, 1000), max_size=100), max_examples=300)
def test_hidden_bug(xs):
    hidden_bug(xs)


@given(text())
def test_rle_roundtrip(s):
    assert rle_decode(rle_encode(s)) == s


@given(integers(0, INT32_MAX), integers(0, INT32_MAX))
def test_midpoint(a, b):
    lo, hi = min(a, b), max(a, b)
    assert lo <= midpoint_int32(lo, hi) <= hi


@given(dictionaries(text("aAbB", max_len=2), integers(0, 100), max_size=6))
def test_merge_keeps_total(d):
    assert sum(merge_case_insensitive(d).values()) == sum(d.values())


@given(integers())
def test_less_than_1000(x):
    assert x < 1000


@given(lists(lists(integers(-50, 50), max_size=5), max_size=8))
def test_reverse_twice(xss):
    assert list(reversed(list(reversed(xss)))) == xss


@given(text(max_len=40))
def test_utf8_roundtrip(s):
    assert s.encode("utf-8").decode("utf-8") == s


class StackMachine(RuleBasedStateMachine):
    """Correct stack: must stay consistent with the model under random push/pop/clear sequences."""

    def __init__(self):
        self.stack, self.model = [], []

    @rule(x=integers(-100, 100))
    def push(self, x):
        self.stack.append(x)
        self.model.append(x)

    @precondition(lambda m: m.model)
    @rule()
    def pop(self):
        assert self.stack.pop() == self.model.pop()

    @rule()
    def clear(self):
        self.stack.clear()
        self.model.clear()

    @invariant()
    def consistent(self):
        assert len(self.stack) == len(self.model)
        assert (self.stack[-1] if self.stack else None) == (self.model[-1] if self.model else None)


class MinStackMachine(RuleBasedStateMachine):
    def __init__(self):
        self.stack, self.model = MinStack(), []

    @rule(x=integers(-100, 100))
    def push(self, x):
        self.stack.push(x)
        self.model.append(x)

    @precondition(lambda m: m.model)
    @rule()
    def pop(self):
        assert self.stack.pop() == self.model.pop()

    @rule()
    def clear(self):
        self.stack.clear()
        self.model.clear()

    @invariant()
    def min_matches_model(self):
        assert self.stack.get_min() == (min(self.model) if self.model else None)


# ───────────── Verification ─────────────

def expect_fail(test, check, label):
    try:
        test()
    except Falsified as f:
        ok = check(f.minimal)
        print(f"       Check         : {'PASSED' if ok else 'FAILED'} — {label}\n")
        return ok
    print(f"       Check         : FAILED — bug not found ({label})\n")
    return False


def expect_pass(test):
    try:
        test()
        return True
    except Falsified:
        return False


def shrink_100_element_input():
    """Shrink a known 100-element input directly (without relying on search luck)."""
    rnd = random.Random(2026)
    big = [rnd.randint(-1000, 1000) for _ in range(100)]

    def fails(xs):
        try:
            hidden_bug(xs)
        except ValueError:
            return True
        return False

    assert fails(big), "the starting input must trigger the bug"
    res = minimize(big, lists(integers(-1000, 1000), max_size=100).shrink, fails)
    print("[FAIL] hidden_bug — fixed 100-element input")
    print(f"       Found input   : {big[:10]}... (length {len(big)})")
    print(f"       Shrinking     : {res.steps} steps accepted / {res.calls} candidates")
    print(f"       Minimal input : {res.value}")
    ok = sorted(res.value) == [-2, 0, 0, 0]
    print(f"       Check         : {'PASSED' if ok else 'FAILED'} — exactly 4 elements, one -2, the rest 0\n")
    return ok


def main():
    print("═" * 72 + "\n PyShrink-PBT — Stage 5 bug scenarios\n" + "═" * 72 + "\n")
    results = [
        shrink_100_element_input(),
        expect_fail(test_hidden_bug, lambda m: sorted(m[0]) == [-2, 0, 0, 0], "[-2, 0, 0, 0] (any order)"),
        expect_fail(test_rle_roundtrip, lambda m: m == ("0",), "simplest text containing a digit: '0'"),
        expect_fail(test_midpoint, lambda m: m[0] + m[1] == INT32_MAX + 1, "lo + hi exactly at the overflow boundary (2^31)"),
        expect_fail(test_merge_keeps_total,
                    lambda m: len(m[0]) == 2 and len({k.lower() for k in m[0]}) == 1
                    and sorted(m[0].values()) == [0, 1], "2 keys colliding by case, values {0, 1}"),
        expect_fail(test_less_than_1000, lambda m: m == (1000,), "exact boundary: 1000"),
        expect_fail(lambda: run_state_machine(MinStackMachine),
                    lambda p: p == [("push", {"x": 0}), ("push", {"x": 0}), ("pop", {})],
                    "push(0), push(0), pop()"),
        expect_pass(test_reverse_twice),
        expect_pass(test_utf8_roundtrip),
        expect_pass(lambda: run_state_machine(StackMachine)),
    ]
    print("═" * 72)
    print(f" RESULT: {sum(results)}/{len(results)} checks passed")
    print("═" * 72)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
