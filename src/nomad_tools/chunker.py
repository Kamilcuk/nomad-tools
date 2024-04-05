from typing import Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


def split_integer(num: int, parts: int) -> List[int]:
    """https://stackoverflow.com/a/58360873/9072753"""
    c, r = divmod(num, parts)
    return [c] * (parts - r) + [c + 1] * r


def splitlist(
    input: List[T], condition: Callable[[T], bool]
) -> Tuple[List[T], List[T]]:
    bad: List[T] = []
    good: List[T] = []
    for x in input:
        (bad, good)[condition(x)].append(x)
    return (bad, good)


def intchunker(input: List[int], lines: int) -> List[int]:
    output: List[Optional[int]] = [None] * len(input)
    chunksize: int = lines // len(input)
    for idx, value in enumerate(input):
        if value >= chunksize:
            output[idx] = value
    lines -= sum(x or 0 for x in output)
    ints: List[int] = split_integer(lines, sum(1 for x in output if x))
    intsidx = 0
    for idx, value in enumerate(input):
        if value < chunksize:
            output[idx] = ints[intsidx]
            intsidx += 1
    ret: List[int] = [x for x in output if x is not None]
    assert len(ret) == len(output)
    return ret


def chunker(input: List[List[T]], lines: int) -> List[T]:
    output: List[T] = []
    chunksize: int = lines // len(input)
    small, high = splitlist(input, lambda values: len(values) <= chunksize)
    for values in small:
        output.extend(values)
    lines -= len(output)
    zip(high, split_integer(lines, len(input) - len(small)))
    for values in high:
        adding = values[-chunksize:]
        output.extend(adding)
        lines -= len(adding)
    return output


if __name__ == "__main__":
    print(intchunker([1, 5, 3, 4], 10))
