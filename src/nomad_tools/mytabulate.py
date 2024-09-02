from typing import List


def mytabulate(data: List[List[str]]) -> str:
    """Print list of list of strings in tabulated form in columns"""
    if not data:
        return ""
    lens: List[int] = [max(len(str(x)) for x in y) for y in data]
    if not lens:
        return ""
    # Last element does not need empty trailing spaces.
    lens[-1] = 0
    rows: int = max(len(x) for x in data)
    cols: int = len(data)
    return "\n".join(
        " ".join(
            "%-*s" % (lens[c], data[c][r] if r < len(data[c]) else "")
            for c in range(cols)
        )
        for r in range(rows)
    )


if __name__ == "__main__":
    print(mytabulate([["a", "bbbbbbb"], ["cc", "dddddd"], ["e", "F", "g"]]))
