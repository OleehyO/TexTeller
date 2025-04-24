import re

from ..utils.latex import change_all
from .format import format_latex


def _rm_dollar_surr(content):
    pattern = re.compile(r"\\[a-zA-Z]+\$.*?\$|\$.*?\$")
    matches = pattern.findall(content)

    for match in matches:
        if not re.match(r"\\[a-zA-Z]+", match):
            new_match = match.strip("$")
            content = content.replace(match, " " + new_match + " ")

    return content


def to_katex(formula: str) -> str:
    """
    Convert LaTeX formula to KaTeX-compatible format.

    This function processes a LaTeX formula string and converts it to a format
    that is compatible with KaTeX rendering. It removes unsupported commands
    and structures, simplifies LaTeX environments, and optimizes the formula
    for web display.

    Args:
        formula: LaTeX formula string to convert

    Returns:
        KaTeX-compatible formula string
    """
    res = formula
    # remove mbox surrounding
    res = change_all(res, r"\mbox ", r" ", r"{", r"}", r"", r"")
    res = change_all(res, r"\mbox", r" ", r"{", r"}", r"", r"")
    # remove hbox surrounding
    res = re.sub(r"\\hbox to ?-? ?\d+\.\d+(pt)?\{", r"\\hbox{", res)
    res = change_all(res, r"\hbox", r" ", r"{", r"}", r"", r" ")
    # remove raise surrounding
    res = re.sub(r"\\raise ?-? ?\d+\.\d+(pt)?", r" ", res)
    # remove makebox
    res = re.sub(r"\\makebox ?\[\d+\.\d+(pt)?\]\{", r"\\makebox{", res)
    res = change_all(res, r"\makebox", r" ", r"{", r"}", r"", r" ")
    # remove vbox surrounding, scalebox surrounding
    res = re.sub(r"\\raisebox\{-? ?\d+\.\d+(pt)?\}\{", r"\\raisebox{", res)
    res = re.sub(r"\\scalebox\{-? ?\d+\.\d+(pt)?\}\{", r"\\scalebox{", res)
    res = change_all(res, r"\scalebox", r" ", r"{", r"}", r"", r" ")
    res = change_all(res, r"\raisebox", r" ", r"{", r"}", r"", r" ")
    res = change_all(res, r"\vbox", r" ", r"{", r"}", r"", r" ")

    origin_instructions = [
        r"\Huge",
        r"\huge",
        r"\LARGE",
        r"\Large",
        r"\large",
        r"\normalsize",
        r"\small",
        r"\footnotesize",
        r"\tiny",
    ]
    for old_ins, new_ins in zip(origin_instructions, origin_instructions):
        res = change_all(res, old_ins, new_ins, r"$", r"$", "{", "}")
    res = change_all(res, r"\mathbf", r"\bm", r"{", r"}", r"{", r"}")
    res = change_all(res, r"\boldmath ", r"\bm", r"{", r"}", r"{", r"}")
    res = change_all(res, r"\boldmath", r"\bm", r"{", r"}", r"{", r"}")
    res = change_all(res, r"\boldmath ", r"\bm", r"$", r"$", r"{", r"}")
    res = change_all(res, r"\boldmath", r"\bm", r"$", r"$", r"{", r"}")
    res = change_all(res, r"\scriptsize", r"\scriptsize", r"$", r"$", r"{", r"}")
    res = change_all(res, r"\emph", r"\textit", r"{", r"}", r"{", r"}")
    res = change_all(res, r"\emph ", r"\textit", r"{", r"}", r"{", r"}")

    # remove bold command
    res = change_all(res, r"\bm", r" ", r"{", r"}", r"", r"")

    origin_instructions = [
        r"\left",
        r"\middle",
        r"\right",
        r"\big",
        r"\Big",
        r"\bigg",
        r"\Bigg",
        r"\bigl",
        r"\Bigl",
        r"\biggl",
        r"\Biggl",
        r"\bigm",
        r"\Bigm",
        r"\biggm",
        r"\Biggm",
        r"\bigr",
        r"\Bigr",
        r"\biggr",
        r"\Biggr",
    ]
    for origin_ins in origin_instructions:
        res = change_all(res, origin_ins, origin_ins, r"{", r"}", r"", r"")

    res = re.sub(r"\\\[(.*?)\\\]", r"\1\\newline", res)

    if res.endswith(r"\newline"):
        res = res[:-8]

    # remove multiple spaces
    res = re.sub(r"(\\,){1,}", " ", res)
    res = re.sub(r"(\\!){1,}", " ", res)
    res = re.sub(r"(\\;){1,}", " ", res)
    res = re.sub(r"(\\:){1,}", " ", res)
    res = re.sub(r"\\vspace\{.*?}", "", res)

    # merge consecutive text
    def merge_texts(match):
        texts = match.group(0)
        merged_content = "".join(re.findall(r"\\text\{([^}]*)\}", texts))
        return f"\\text{{{merged_content}}}"

    res = re.sub(r"(\\text\{[^}]*\}\s*){2,}", merge_texts, res)

    res = res.replace(r"\bf ", "")
    res = _rm_dollar_surr(res)

    # remove extra spaces (keeping only one)
    res = re.sub(r" +", " ", res)

    # format latex
    res = res.strip()
    res = format_latex(res)

    return res
