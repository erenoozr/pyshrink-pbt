"""PyShrink-PBT: a property-based testing and shrinking engine from scratch (standard library only).

Stage 1  Strategies       : integers, text, lists, dictionaries, tuples
Stage 2  @given           : run with random examples, stop at the first failure, record the input
Stage 3  Shrinking        : greedy delta debugging, deterministic
Stage 4  State machines   : RuleBasedStateMachine + @rule / @invariant / @precondition
"""
from __future__ import annotations

import copy
import functools
import inspect
import random
import string
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterator

__all__ = [
    "Strategy", "integers", "text", "lists", "dictionaries", "tuples",
    "given", "Falsified", "minimize", "ShrinkResult",
    "RuleBasedStateMachine", "rule", "invariant", "precondition", "run_state_machine",
]

DEFAULT_MAX_SHRINKS = 5000

# Order = simplicity order: the shrinker pulls characters towards the start of this string.
DEFAULT_ALPHABET = (
    string.ascii_lowercase + string.digits + string.ascii_uppercase
    + " .,;:!?-_()[]{}<>/@#$%^&*+=~|`"
    + "\t\n\r\\\"'\x00\x1b"
    + "çğıöşüÇĞİÖŞÜé€✓漢字😀​́"
)


# ═════════════════════════ Stage 1: Strategies ═════════════════════════

class Strategy:
    """Generator + shrinker. shrink() yields candidates simplest first; every candidate is strictly simpler than its input."""

    def generate(self, rnd: random.Random) -> Any:
        raise NotImplementedError

    def shrink(self, value: Any) -> Iterator[Any]:
        raise NotImplementedError


def _half(d: int) -> int:
    """Integer halving rounded towards zero (no floats, safe for big ints)."""
    return d // 2 if d >= 0 else -((-d) // 2)


def _shrink_seq(seq: list, shrink_elem: Callable[[Any], Iterator[Any]], min_size: int = 0) -> Iterator[list]:
    """Sequence shrinking: empty → split in half → delete chunks (n/2 … 1) → shrink equal elements together → shrink each element."""
    n = len(seq)
    if n > min_size:
        yield seq[:min_size]
        half = n // 2
        if min_size < half:
            yield seq[:half]
        if half and n - half >= min_size:
            yield seq[half:]
        k = half
        while k >= 1:
            if n - k >= min_size:
                for i in range(0, n - k + 1, k):
                    yield seq[:i] + seq[i + k:]
            k //= 2
    # Shrinking equal elements together escapes "two steps need the same value" local minima.
    for i, x in enumerate(seq):
        if any(y == x for y in seq[:i]) or not any(y == x for y in seq[i + 1:]):
            continue
        for c in shrink_elem(x):
            yield [c if y == x else y for y in seq]
    for i, x in enumerate(seq):
        for c in shrink_elem(x):
            yield seq[:i] + [c] + seq[i + 1:]


class _Integers(Strategy):
    def __init__(self, lo: int, hi: int):
        if lo > hi:
            raise ValueError(f"integers: min_val ({lo}) > max_val ({hi})")
        self.lo, self.hi = lo, hi
        self.target = min(max(0, lo), hi)  # value in range closest to zero
        edges = {0, 1, -1, 2, -2, lo, hi, lo + 1, hi - 1}
        for b in (7, 8, 15, 16, 31, 32, 63, 64):
            edges |= {2**b - 1, 2**b, -(2**b), -(2**b) - 1}
        self.edges = sorted(e for e in edges if lo <= e <= hi)

    def generate(self, rnd: random.Random) -> int:
        r = rnd.random()
        if r < 0.3:
            return rnd.choice(self.edges)
        a, b = max(self.lo, -16), min(self.hi, 16)
        if r < 0.55 and a <= b:
            return rnd.randint(a, b)
        return rnd.randint(self.lo, self.hi)

    def shrink(self, v: int) -> Iterator[int]:
        t = self.target
        if v == t:
            return
        yield t
        if v < 0 and -v <= self.hi:
            yield -v  # a positive value of the same magnitude is simpler
        d = _half(v - t)
        while d:  # binary search: halfway to t, quarter way, … one step from v
            yield v - d
            d = _half(d)
        dist = abs(v - t)
        for m in range(1, min(dist, 33)):  # small-value probe: catches values like -2 or 4 that binary steps skip
            for c in (t + m, t - m):
                if self.lo <= c <= self.hi:
                    yield c


def integers(min_val: int = -(2**63), max_val: int = 2**63 - 1) -> Strategy:
    """Integer strategy biased towards edge values (0, ±1, bounds, 2^k boundaries)."""
    return _Integers(min_val, max_val)


class _Text(Strategy):
    def __init__(self, alphabet: str, min_len: int, max_len: int):
        if not alphabet:
            raise ValueError("text: alphabet must not be empty")
        if not 0 <= min_len <= max_len:
            raise ValueError("text: requires 0 <= min_len <= max_len")
        self.alphabet = "".join(dict.fromkeys(alphabet))  # deduplicated, order preserved
        self.rank = {c: i for i, c in enumerate(self.alphabet)}
        self.tricky = [c for c in self.alphabet if not (c.isascii() and c.isalnum())]
        self.min_len, self.max_len = min_len, max_len

    def _char(self, rnd: random.Random) -> str:
        if self.tricky and rnd.random() < 0.2:
            return rnd.choice(self.tricky)  # favour escape and unicode characters
        return rnd.choice(self.alphabet)

    def generate(self, rnd: random.Random) -> str:
        if self.min_len == 0 and rnd.random() < 0.1:
            return ""
        return "".join(self._char(rnd) for _ in range(rnd.randint(self.min_len, self.max_len)))

    def _shrink_char(self, c: str) -> Iterator[str]:
        i = self.rank[c]
        if i == 0:
            return
        yield self.alphabet[0]
        d = i // 2
        while d:
            yield self.alphabet[i - d]
            d //= 2

    def shrink(self, s: str) -> Iterator[str]:
        for chars in _shrink_seq(list(s), self._shrink_char, self.min_len):
            yield "".join(chars)


def text(alphabet: str = DEFAULT_ALPHABET, min_len: int = 0, max_len: int = 20) -> Strategy:
    """Text strategy. Alphabet order is simplicity order (default: a-z, 0-9, A-Z, punctuation, escapes, unicode)."""
    return _Text(alphabet, min_len, max_len)


class _Lists(Strategy):
    def __init__(self, elements: Strategy, min_size: int, max_size: int):
        if not 0 <= min_size <= max_size:
            raise ValueError("lists: requires 0 <= min_size <= max_size")
        self.elements, self.min_size, self.max_size = elements, min_size, max_size

    def generate(self, rnd: random.Random) -> list:
        r = rnd.random()
        n = self.min_size if r < 0.1 else self.max_size if r < 0.2 else rnd.randint(self.min_size, self.max_size)
        return [self.elements.generate(rnd) for _ in range(n)]

    def shrink(self, xs: list) -> Iterator[list]:
        return _shrink_seq(xs, self.elements.shrink, self.min_size)


def lists(elements: Strategy, min_size: int = 0, max_size: int = 10) -> Strategy:
    """List strategy; elements may itself be lists(...) (nested lists)."""
    return _Lists(elements, min_size, max_size)


class _Dicts(Strategy):
    def __init__(self, keys: Strategy, values: Strategy, min_size: int, max_size: int):
        if not 0 <= min_size <= max_size:
            raise ValueError("dictionaries: requires 0 <= min_size <= max_size")
        self.keys, self.values, self.min_size, self.max_size = keys, values, min_size, max_size

    def generate(self, rnd: random.Random) -> dict:
        n = rnd.randint(self.min_size, self.max_size)
        out: dict = {}
        for _ in range(n * 10):  # bounded attempts to find unique keys
            if len(out) >= n:
                break
            k = self.keys.generate(rnd)
            if k not in out:
                out[k] = self.values.generate(rnd)
        if len(out) < self.min_size:
            raise ValueError("dictionaries: key space too small for min_size")
        return out

    def _shrink_item(self, item: tuple) -> Iterator[tuple]:
        k, v = item
        for k2 in self.keys.shrink(k):
            yield k2, v
        for v2 in self.values.shrink(v):
            yield k, v2

    def shrink(self, d: dict) -> Iterator[dict]:
        for items in _shrink_seq(list(d.items()), self._shrink_item, self.min_size):
            cand = dict(items)
            if len(cand) == len(items):  # skip candidates with colliding keys
                yield cand


def dictionaries(keys: Strategy, values: Strategy, min_size: int = 0, max_size: int = 10) -> Strategy:
    return _Dicts(keys, values, min_size, max_size)


class _Tuples(Strategy):
    def __init__(self, parts: tuple[Strategy, ...]):
        self.parts = parts

    def generate(self, rnd: random.Random) -> tuple:
        return tuple(s.generate(rnd) for s in self.parts)

    def shrink(self, v: tuple) -> Iterator[tuple]:
        for i, s in enumerate(self.parts):
            for c in s.shrink(v[i]):
                yield v[:i] + (c,) + v[i + 1:]


def tuples(*parts: Strategy) -> Strategy:
    for s in parts:
        if not isinstance(s, Strategy):
            raise TypeError(f"expected a Strategy, got {type(s).__name__}")
    return _Tuples(parts)


# ═════════════════════════ Stage 3: Shrinking ═════════════════════════

@dataclass
class ShrinkResult:
    value: Any
    steps: int       # accepted shrink steps
    calls: int       # candidates tried
    exhausted: bool  # True if the budget ran out (result still fails but may not be minimal)


def minimize(value: Any, shrink_fn: Callable[[Any], Iterator[Any]],
             still_fails: Callable[[Any], bool], max_calls: int = DEFAULT_MAX_SHRINKS) -> ShrinkResult:
    """Greedy shrinking: accept the first candidate that still fails, restart; stop when no candidate fails."""
    steps = calls = 0
    improved = True
    while improved:
        improved = False
        for cand in shrink_fn(value):
            if calls >= max_calls:
                return ShrinkResult(value, steps, calls, True)
            calls += 1
            if still_fails(cand):
                value, steps, improved = cand, steps + 1, True
                break
    return ShrinkResult(value, steps, calls, False)


# ═════════════════════════ Stage 2: @given ═════════════════════════

class Falsified(AssertionError):
    """A property was falsified. original: the first failing input, minimal: the shrunk input."""

    def __init__(self, message: str, original: Any, minimal: Any, cause: BaseException | None):
        super().__init__(message)
        self.original, self.minimal, self.cause = original, minimal, cause


def _run(fn: Callable, args: tuple) -> Exception | None:
    try:
        fn(*copy.deepcopy(args))  # shrinking stays correct even if the test mutates its input
    except Exception as e:
        return e
    return None


def _fmt(v: Any, limit: int = 160) -> str:
    s = repr(v)
    if len(s) > limit:
        s = s[:limit - 3] + "..."
    if isinstance(v, (list, str, dict, tuple)):
        s += f"   (length {len(v)})"
    return s


def _out(msg: str) -> None:
    enc = sys.stdout.encoding or "utf-8"
    print(msg.encode(enc, "backslashreplace").decode(enc))  # don't crash on emoji etc. in legacy consoles


def _err_line(e: BaseException | None) -> str:
    return f"{type(e).__name__}: {e}" if e is not None else "(no error)"


def given(*strategies: Strategy, max_examples: int = 200, seed: int | None = None,
          max_shrinks: int = DEFAULT_MAX_SHRINKS) -> Callable:
    """Runs the test with max_examples random inputs; on failure shrinks the input and raises Falsified."""
    args_strategy = tuples(*strategies)

    def decorator(fn: Callable) -> Callable:
        params = list(inspect.signature(fn).parameters)
        if len(params) != len(strategies):
            raise TypeError(f"{fn.__name__}: has {len(params)} parameters but {len(strategies)} strategies were given")

        @functools.wraps(fn)
        def wrapper() -> None:
            run_seed = random.randrange(2**32) if seed is None else seed
            rnd = random.Random(run_seed)
            for i in range(1, max_examples + 1):
                args = args_strategy.generate(rnd)
                err = _run(fn, args)
                if err is not None:
                    break
            else:
                _out(f"[ OK ] {fn.__name__}: {max_examples} examples passed (seed={run_seed})")
                return
            kind = type(err)  # only accept candidates with the same error type (don't slide into another bug)
            res = minimize(args, args_strategy.shrink, lambda a: type(_run(fn, a)) is kind, max_shrinks)
            final = _run(fn, res.value)
            show = lambda a: ", ".join(f"{p}={_fmt(x)}" for p, x in zip(params, a))
            report = "\n".join([
                f"[FAIL] {fn.__name__}  (seed={run_seed}, example {i}/{max_examples})",
                f"       Found input   : {show(args)}",
                f"       Shrinking     : {res.steps} steps accepted / {res.calls} candidates"
                + ("  [budget exhausted]" if res.exhausted else ""),
                f"       Minimal input : {show(res.value)}",
                f"       Error         : {_err_line(final)}",
            ])
            _out(report)
            raise Falsified(report, args, res.value, final) from final

        wrapper.__signature__ = inspect.Signature()  # keep pytest from treating parameters as fixtures
        return wrapper

    return decorator


# ═════════════════════════ Stage 4: State machines ═════════════════════════

def rule(**strategies: Strategy) -> Callable:
    """Marks a machine method as a rule; keyword arguments are drawn from the strategies."""
    def deco(fn: Callable) -> Callable:
        fn._pbt_rule = strategies
        return fn
    return deco


def invariant() -> Callable:
    """Check run after every step (and at start)."""
    def deco(fn: Callable) -> Callable:
        fn._pbt_invariant = True
        return fn
    return deco


def precondition(pred: Callable[[Any], bool]) -> Callable:
    """The rule is only chosen/replayed while pred(machine) is True. Works in either order with @rule."""
    def deco(fn: Callable) -> Callable:
        fn._pbt_precondition = pred
        return fn
    return deco


class RuleBasedStateMachine:
    """Set up state in a subclass __init__; define methods with @rule, @invariant, @precondition."""

    def teardown(self) -> None:
        """Called at the end of every run; override if resources need cleaning up."""


Step = tuple  # (rule_name, {argument: value})


def _enabled(fn: Callable, machine: Any) -> bool:
    pre = getattr(fn, "_pbt_precondition", None)
    return pre is None or bool(pre(machine))


def _collect(cls: type) -> tuple[dict[str, Callable], list[tuple[str, Callable]]]:
    rules: dict[str, Callable] = {}
    invs: list[tuple[str, Callable]] = []
    for name in sorted(dir(cls)):  # sorted: the same seed produces the same program
        fn = getattr(cls, name, None)
        if hasattr(fn, "_pbt_rule"):
            rules[name] = fn
        elif getattr(fn, "_pbt_invariant", False):
            invs.append((name, fn))
    if not rules:
        raise TypeError(f"{cls.__name__}: no @rule defined")
    return rules, invs


def _play(cls: type, rules: dict, invs: list, choose: Callable[[Any], Step | None]):
    """Builds the machine and runs the steps from choose(). Returns (error, location, executed_steps)."""
    executed: list[Step] = []
    where, err, machine = "__init__", None, None
    try:
        machine = cls()
        for inv_name, inv in invs:
            where = f"invariant {inv_name} (initial)"
            inv(machine)
        while (step := choose(machine)) is not None:
            name, kw = step
            executed.append(step)
            where = f"rule {name}"
            rules[name](machine, **copy.deepcopy(kw))
            for inv_name, inv in invs:
                where = f"invariant {inv_name} (after {name})"
                inv(machine)
    except Exception as e:
        err = e
    finally:
        if machine is not None:
            machine.teardown()
    return err, where, executed


def run_state_machine(cls: type, max_examples: int = 100, max_steps: int = 30, seed: int | None = None,
                      max_shrinks: int = DEFAULT_MAX_SHRINKS) -> None:
    """Runs random rule sequences; on inconsistency shrinks the step program and raises Falsified."""
    rules, invs = _collect(cls)
    run_seed = random.randrange(2**32) if seed is None else seed
    rnd = random.Random(run_seed)

    def generator(n: int) -> Callable:
        count = 0
        pool: dict[tuple[str, str], list] = {}

        def arg(name: str, key: str, strat: Strategy) -> Any:
            seen = pool.setdefault((name, key), [])
            if seen and rnd.random() < 0.25:
                return rnd.choice(seen)  # repeated values (the same push twice) are a common bug trigger
            seen.append(strat.generate(rnd))
            return seen[-1]

        def choose(m: Any) -> Step | None:
            nonlocal count
            enabled = [name for name, fn in rules.items() if _enabled(fn, m)]
            if count >= n or not enabled:
                return None
            count += 1
            name = rnd.choice(enabled)
            return name, {k: arg(name, k, s) for k, s in rules[name]._pbt_rule.items()}
        return choose

    def replayer(program: list[Step]) -> Callable:
        it = iter(program)

        def choose(m: Any) -> Step | None:
            for name, kw in it:  # steps whose precondition no longer holds are skipped
                if _enabled(rules[name], m):
                    return name, kw
            return None
        return choose

    for i in range(1, max_examples + 1):
        err, where, program = _play(cls, rules, invs, generator(rnd.randint(1, max_steps)))
        if err is not None:
            break
    else:
        _out(f"[ OK ] {cls.__name__}: {max_examples} scenarios (≤{max_steps} steps) consistent (seed={run_seed})")
        return

    def shrink_step(step: Step) -> Iterator[Step]:
        name, kw = step
        for key, strat in rules[name]._pbt_rule.items():
            for c in strat.shrink(kw[key]):
                yield name, {**kw, key: c}

    kind = type(err)
    res = minimize(program, lambda p: _shrink_seq(p, shrink_step),
                   lambda p: type(_play(cls, rules, invs, replayer(p))[0]) is kind, max_shrinks)
    final, where, executed = _play(cls, rules, invs, replayer(res.value))
    lines = [
        f"[FAIL] {cls.__name__}  (seed={run_seed}, scenario {i}/{max_examples})",
        f"       Found program   : {len(program)} steps → shrunk to: {len(executed)} steps "
        f"({res.steps} steps accepted / {res.calls} candidates" + (", budget exhausted)" if res.exhausted else ")"),
        f"       state = {cls.__name__}()",
        *(f"       state.{n}({', '.join(f'{k}={v!r}' for k, v in kw.items())})" for n, kw in executed),
        f"       Error location  : {where}",
        f"       Error           : {_err_line(final)}",
    ]
    report = "\n".join(lines)
    _out(report)
    raise Falsified(report, program, executed, final) from final
