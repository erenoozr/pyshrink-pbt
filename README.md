# PyShrink-PBT

A property-based testing and shrinking engine written from scratch in pure Python (3.10+, standard library only).

PyShrink generates hundreds of random inputs for a property, stops at the first failure, and then **shrinks** the failing input deterministically down to a minimal counterexample, e.g. a random 100-element list becomes `[-2, 0, 0, 0]`.

## Features

- **Strategies:** `integers`, `text`, `lists` (nestable), `dictionaries`, `tuples`
  - Integers are biased towards edge cases: `0`, `±1`, range bounds and power-of-two boundaries (`127/128`, `2^31`, `2^63`, …).
  - Text is biased towards empty strings, escape characters (`\n`, `\t`, `\0`, `\\`, quotes) and unicode/emoji.
- **`@given` decorator:** runs a test with many random inputs and stops at the first `AssertionError` or unexpected exception. It then raises `Falsified`, which carries both the original and the minimal input.
- **Greedy delta-debugging shrinker:**
  - Lists: try emptying, then splitting in half, then deleting chunks of size n/2 … 1.
  - Integers: binary-search towards zero, plus a small-value probe.
  - Strings: shortened, then each character simplified towards the start of the alphabet.
  - Equal elements are shrunk together to escape local minima.
  - Only candidates that fail with the *same* exception type are accepted, so shrinking never slides into a different bug.
- **Stateful testing:** `RuleBasedStateMachine` with `@rule`, `@invariant` and `@precondition`. Failing step sequences are shrunk and printed as a replayable program.

## Usage

```python
from pyshrink import given, integers, lists

@given(lists(integers(-1000, 1000), max_size=100))
def test_no_negative_even(xs):
    if len(xs) > 3 and any(x < 0 and x % 2 == 0 for x in xs):
        raise ValueError("hidden edge case")

test_no_negative_even()   # raises Falsified, minimal input: xs=[0, 0, 0, -2]
```

### Stateful testing

```python
from pyshrink import RuleBasedStateMachine, rule, invariant, precondition, integers, run_state_machine

class StackMachine(RuleBasedStateMachine):
    def __init__(self):
        self.stack, self.model = MyStack(), []

    @rule(x=integers(-100, 100))
    def push(self, x):
        self.stack.push(x); self.model.append(x)

    @precondition(lambda m: m.model)
    @rule()
    def pop(self):
        assert self.stack.pop() == self.model.pop()

    @invariant()
    def consistent(self):
        assert len(self.stack) == len(self.model)

run_state_machine(StackMachine)
```

## Demo

```
python demo.py
```

`demo.py` defines functions with deliberately hidden bugs and checks that PyShrink finds and minimizes each one:

| Hidden bug | Minimal counterexample |
|---|---|
| Crashes if length > 3 and a negative even number is present | `[-2, 0, 0, 0]` (from a 100-element list) |
| Run-length encoder breaks on digits in the input | `'0'` |
| int32 midpoint overflow (`lo + hi`) | a pair summing to exactly `2^31` |
| Case-insensitive merge overwrites instead of summing | `{'a': 1, 'A': 0}` |
| `x < 1000` | `1000` |
| Min-stack uses `<` instead of `<=` | `push(0), push(0), pop()` |

Passing properties (double reverse, UTF-8 round trip, a correct stack machine) are included as well. The script exits with code `0` when every check passes.
