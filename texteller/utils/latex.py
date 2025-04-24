import re


def _change(input_str, old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r):
    result = ""
    i = 0
    n = len(input_str)

    while i < n:
        if input_str[i : i + len(old_inst)] == old_inst:
            # check if the old_inst is followed by old_surr_l
            start = i + len(old_inst)
        else:
            result += input_str[i]
            i += 1
            continue

        if start < n and input_str[start] == old_surr_l:
            # found an old_inst followed by old_surr_l, now look for the matching old_surr_r
            count = 1
            j = start + 1
            escaped = False
            while j < n and count > 0:
                if input_str[j] == "\\" and not escaped:
                    escaped = True
                    j += 1
                    continue
                if input_str[j] == old_surr_r and not escaped:
                    count -= 1
                    if count == 0:
                        break
                elif input_str[j] == old_surr_l and not escaped:
                    count += 1
                escaped = False
                j += 1

            if count == 0:
                assert j < n
                assert input_str[start] == old_surr_l
                assert input_str[j] == old_surr_r
                inner_content = input_str[start + 1 : j]
                # Replace the content with new pattern
                result += new_inst + new_surr_l + inner_content + new_surr_r
                i = j + 1
                continue
            else:
                assert count >= 1
                assert j == n
                print("Warning: unbalanced surrogate pair in input string")
                result += new_inst + new_surr_l
                i = start + 1
                continue
        else:
            result += input_str[i:start]
            i = start

    if old_inst != new_inst and (old_inst + old_surr_l) in result:
        return _change(result, old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r)
    else:
        return result


def _find_substring_positions(string, substring):
    positions = [match.start() for match in re.finditer(re.escape(substring), string)]
    return positions


def change_all(input_str, old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r):
    pos = _find_substring_positions(input_str, old_inst + old_surr_l)
    res = list(input_str)
    for p in pos[::-1]:
        res[p:] = list(
            _change(
                "".join(res[p:]), old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r
            )
        )
    res = "".join(res)
    return res


def remove_style(input_str: str) -> str:
    input_str = change_all(input_str, r"\bm", r" ", r"{", r"}", r"", r" ")
    input_str = change_all(input_str, r"\boldsymbol", r" ", r"{", r"}", r"", r" ")
    input_str = change_all(input_str, r"\textit", r" ", r"{", r"}", r"", r" ")
    input_str = change_all(input_str, r"\textbf", r" ", r"{", r"}", r"", r" ")
    input_str = change_all(input_str, r"\textbf", r" ", r"{", r"}", r"", r" ")
    input_str = change_all(input_str, r"\mathbf", r" ", r"{", r"}", r"", r" ")
    output_str = input_str.strip()
    return output_str


def add_newlines(latex_str: str) -> str:
    """
    Adds newlines to a LaTeX string based on specific patterns, ensuring no
    duplicate newlines are added around begin/end environments.
    - After \\ (if not already followed by newline)
    - Before \\begin{...} (if not already preceded by newline)
    - After \\begin{...} (if not already followed by newline)
    - Before \\end{...} (if not already preceded by newline)
    - After \\end{...} (if not already followed by newline)

    Args:
        latex_str: The input LaTeX string.

    Returns:
        The LaTeX string with added newlines, avoiding duplicates.
    """
    processed_str = latex_str

    # 1. Replace whitespace around \begin{...} with \n...\n
    # \s* matches zero or more whitespace characters (space, tab, newline)
    # Captures the \begin{...} part in group 1 (\g<1>)
    processed_str = re.sub(r"\s*(\\begin\{[^}]*\})\s*", r"\n\g<1>\n", processed_str)

    # 2. Replace whitespace around \end{...} with \n...\n
    # Same logic as for \begin
    processed_str = re.sub(r"\s*(\\end\{[^}]*\})\s*", r"\n\g<1>\n", processed_str)

    # 3. Add newline after \\ (if not already followed by newline)
    processed_str = re.sub(r"\\\\(?!\n| )|\\\\ ", r"\\\\\n", processed_str)

    # 4. Cleanup: Collapse multiple consecutive newlines into a single newline.
    # This handles cases where the replacements above might have created \n\n.
    processed_str = re.sub(r"\n{2,}", "\n", processed_str)

    # Remove leading/trailing whitespace (including potential single newlines
    # at the very start/end resulting from the replacements) from the entire result.
    return processed_str.strip()
