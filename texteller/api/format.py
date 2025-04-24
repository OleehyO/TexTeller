#!/usr/bin/env python3
"""
Python implementation of tex-fmt, a LaTeX formatter.
Based on the Rust implementation at https://github.com/WGUNDERWOOD/tex-fmt
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Constants
LINE_END = "\n"
ITEM = "\\item"
DOC_BEGIN = "\\begin{document}"
DOC_END = "\\end{document}"
ENV_BEGIN = "\\begin{"
ENV_END = "\\end{"
TEXT_LINE_START = ""
COMMENT_LINE_START = "% "

# Opening and closing delimiters
OPENS = ["{", "(", "["]
CLOSES = ["}", ")", "]"]

# Names of LaTeX verbatim environments
VERBATIMS = ["verbatim", "Verbatim", "lstlisting", "minted", "comment"]
VERBATIMS_BEGIN = [f"\\begin{{{v}}}" for v in VERBATIMS]
VERBATIMS_END = [f"\\end{{{v}}}" for v in VERBATIMS]

# Regex patterns for sectioning commands
SPLITTING = [
    r"\\begin\{",
    r"\\end\{",
    r"\\item(?:$|[^a-zA-Z])",
    r"\\(?:sub){0,2}section\*?\{",
    r"\\chapter\*?\{",
    r"\\part\*?\{",
]

# Compiled regexes
SPLITTING_STRING = f"({'|'.join(SPLITTING)})"
RE_NEWLINES = re.compile(f"{LINE_END}{LINE_END}({LINE_END})+")
RE_TRAIL = re.compile(f" +{LINE_END}")
RE_SPLITTING = re.compile(SPLITTING_STRING)
RE_SPLITTING_SHARED_LINE = re.compile(f"(?:\\S.*?)(?:{SPLITTING_STRING}.*)")
RE_SPLITTING_SHARED_LINE_CAPTURE = re.compile(f"(?P<prev>\\S.*?)(?P<env>{SPLITTING_STRING}.*)")


@dataclass
class Args:
    """Formatter configuration."""

    tabchar: str = " "
    tabsize: int = 4
    wrap: bool = False
    wraplen: int = 80
    wrapmin: int = 40
    lists: List[str] = None
    verbosity: int = 0

    def __post_init__(self):
        if self.lists is None:
            self.lists = []


@dataclass
class Ignore:
    """Information on the ignored state of a line."""

    actual: bool = False
    visual: bool = False

    @classmethod
    def new(cls):
        return cls(False, False)


@dataclass
class Verbatim:
    """Information on the verbatim state of a line."""

    actual: int = 0
    visual: bool = False

    @classmethod
    def new(cls):
        return cls(0, False)


@dataclass
class Indent:
    """Information on the indentation state of a line."""

    actual: int = 0
    visual: int = 0

    @classmethod
    def new(cls):
        return cls(0, 0)


@dataclass
class State:
    """Information on the current state during formatting."""

    linum_old: int = 1
    linum_new: int = 1
    ignore: Ignore = None
    indent: Indent = None
    verbatim: Verbatim = None
    linum_last_zero_indent: int = 1

    def __post_init__(self):
        if self.ignore is None:
            self.ignore = Ignore.new()
        if self.indent is None:
            self.indent = Indent.new()
        if self.verbatim is None:
            self.verbatim = Verbatim.new()


@dataclass
class Pattern:
    """Record whether a line contains certain patterns."""

    contains_env_begin: bool = False
    contains_env_end: bool = False
    contains_item: bool = False
    contains_splitting: bool = False
    contains_comment: bool = False

    @classmethod
    def new(cls, s: str):
        """Check if a string contains patterns."""
        if RE_SPLITTING.search(s):
            return cls(
                contains_env_begin=ENV_BEGIN in s,
                contains_env_end=ENV_END in s,
                contains_item=ITEM in s,
                contains_splitting=True,
                contains_comment="%" in s,
            )
        else:
            return cls(
                contains_env_begin=False,
                contains_env_end=False,
                contains_item=False,
                contains_splitting=False,
                contains_comment="%" in s,
            )


@dataclass
class Log:
    """Log message."""

    level: str
    file: str
    message: str
    linum_new: Optional[int] = None
    linum_old: Optional[int] = None
    line: Optional[str] = None


def find_comment_index(line: str, pattern: Pattern) -> Optional[int]:
    """Find the index of a comment in a line."""
    if not pattern.contains_comment:
        return None

    in_command = False
    for i, c in enumerate(line):
        if c == "\\":
            in_command = True
        elif in_command and not c.isalpha():
            in_command = False
        elif c == "%" and not in_command:
            return i

    return None


def contains_ignore_skip(line: str) -> bool:
    """Check if a line contains a skip directive."""
    return line.endswith("% tex-fmt: skip")


def contains_ignore_begin(line: str) -> bool:
    """Check if a line contains the start of an ignore block."""
    return line.endswith("% tex-fmt: off")


def contains_ignore_end(line: str) -> bool:
    """Check if a line contains the end of an ignore block."""
    return line.endswith("% tex-fmt: on")


def get_ignore(line: str, state: State, logs: List[Log], file: str, warn: bool) -> Ignore:
    """Determine whether a line should be ignored."""
    skip = contains_ignore_skip(line)
    begin = contains_ignore_begin(line)
    end = contains_ignore_end(line)

    if skip:
        actual = state.ignore.actual
        visual = True
    elif begin:
        actual = True
        visual = True
        if warn and state.ignore.actual:
            logs.append(
                Log(
                    level="WARN",
                    file=file,
                    message="Cannot begin ignore block:",
                    linum_new=state.linum_new,
                    linum_old=state.linum_old,
                    line=line,
                )
            )
    elif end:
        actual = False
        visual = True
        if warn and not state.ignore.actual:
            logs.append(
                Log(
                    level="WARN",
                    file=file,
                    message="No ignore block to end.",
                    linum_new=state.linum_new,
                    linum_old=state.linum_old,
                    line=line,
                )
            )
    else:
        actual = state.ignore.actual
        visual = state.ignore.actual

    return Ignore(actual=actual, visual=visual)


def get_verbatim_diff(line: str, pattern: Pattern) -> int:
    """Calculate total verbatim depth change."""
    if pattern.contains_env_begin and any(r in line for r in VERBATIMS_BEGIN):
        return 1
    elif pattern.contains_env_end and any(r in line for r in VERBATIMS_END):
        return -1
    else:
        return 0


def get_verbatim(
    line: str, state: State, logs: List[Log], file: str, warn: bool, pattern: Pattern
) -> Verbatim:
    """Determine whether a line is in a verbatim environment."""
    diff = get_verbatim_diff(line, pattern)
    actual = state.verbatim.actual + diff
    visual = actual > 0 or state.verbatim.actual > 0

    if warn and actual < 0:
        logs.append(
            Log(
                level="WARN",
                file=file,
                message="Verbatim count is negative.",
                linum_new=state.linum_new,
                linum_old=state.linum_old,
                line=line,
            )
        )

    return Verbatim(actual=actual, visual=visual)


def get_diff(line: str, pattern: Pattern, lists_begin: List[str], lists_end: List[str]) -> int:
    """Calculate total indentation change due to the current line."""
    diff = 0

    # Other environments get single indents
    if pattern.contains_env_begin and ENV_BEGIN in line:
        # Documents get no global indentation
        if DOC_BEGIN in line:
            return 0
        diff += 1
        diff += 1 if any(r in line for r in lists_begin) else 0
    elif pattern.contains_env_end and ENV_END in line:
        # Documents get no global indentation
        if DOC_END in line:
            return 0
        diff -= 1
        diff -= 1 if any(r in line for r in lists_end) else 0

    # Indent for delimiters
    for c in line:
        if c in OPENS:
            diff += 1
        elif c in CLOSES:
            diff -= 1

    return diff


def get_back(line: str, pattern: Pattern, state: State, lists_end: List[str]) -> int:
    """Calculate dedentation for the current line."""
    # Only need to dedent if indentation is present
    if state.indent.actual == 0:
        return 0

    if pattern.contains_env_end and ENV_END in line:
        # Documents get no global indentation
        if DOC_END in line:
            return 0
        # List environments get double indents for indenting items
        for r in lists_end:
            if r in line:
                return 2
        return 1

    # Items get dedented
    if pattern.contains_item and ITEM in line:
        return 1

    return 0


def get_indent(
    line: str,
    prev_indent: Indent,
    pattern: Pattern,
    state: State,
    lists_begin: List[str],
    lists_end: List[str],
) -> Indent:
    """Calculate the indent for a line."""
    diff = get_diff(line, pattern, lists_begin, lists_end)
    back = get_back(line, pattern, state, lists_end)

    actual = prev_indent.actual + diff
    visual = max(0, prev_indent.actual - back)

    return Indent(actual=actual, visual=visual)


def calculate_indent(
    line: str,
    state: State,
    logs: List[Log],
    file: str,
    args: Args,
    pattern: Pattern,
    lists_begin: List[str],
    lists_end: List[str],
) -> Indent:
    """Calculate the indent for a line and update the state."""
    indent = get_indent(line, state.indent, pattern, state, lists_begin, lists_end)

    # Update the state
    state.indent = indent

    # Record the last line with zero indent
    if indent.visual == 0:
        state.linum_last_zero_indent = state.linum_new

    return indent


def apply_indent(line: str, indent: Indent, args: Args, indent_char: str) -> str:
    """Apply indentation to a line."""
    if not line.strip():
        return ""

    indent_str = indent_char * (indent.visual * args.tabsize)
    return indent_str + line.lstrip()


def needs_wrap(line: str, indent_length: int, args: Args) -> bool:
    """Check if a line needs wrapping."""
    return args.wrap and (len(line) + indent_length > args.wraplen)


def find_wrap_point(line: str, indent_length: int, args: Args) -> Optional[int]:
    """Find the best place to break a long line."""
    wrap_point = None
    after_char = False
    prev_char = None

    line_width = 0
    wrap_boundary = args.wrapmin - indent_length

    for i, c in enumerate(line):
        line_width += 1
        if line_width > wrap_boundary and wrap_point is not None:
            break
        if c == " " and prev_char != "\\":
            if after_char:
                wrap_point = i
        elif c != "%":
            after_char = True
        prev_char = c

    return wrap_point


def apply_wrap(
    line: str,
    indent_length: int,
    state: State,
    file: str,
    args: Args,
    logs: List[Log],
    pattern: Pattern,
) -> Optional[List[str]]:
    """Wrap a long line into a short prefix and a suffix."""
    if args.verbosity >= 3:  # Trace level
        logs.append(
            Log(
                level="TRACE",
                file=file,
                message="Wrapping long line.",
                linum_new=state.linum_new,
                linum_old=state.linum_old,
                line=line,
            )
        )

    wrap_point = find_wrap_point(line, indent_length, args)
    comment_index = find_comment_index(line, pattern)

    if wrap_point is None or wrap_point > args.wraplen:
        logs.append(
            Log(
                level="WARN",
                file=file,
                message="Line cannot be wrapped.",
                linum_new=state.linum_new,
                linum_old=state.linum_old,
                line=line,
            )
        )
        return None

    this_line = line[:wrap_point]

    if comment_index is not None and wrap_point > comment_index:
        next_line_start = COMMENT_LINE_START
    else:
        next_line_start = TEXT_LINE_START

    next_line = line[wrap_point + 1 :]

    return [this_line, next_line_start, next_line]


def needs_split(line: str, pattern: Pattern) -> bool:
    """Check if line contains content which should be split onto a new line."""
    # Check if we should format this line and if we've matched an environment
    contains_splittable_env = (
        pattern.contains_splitting and RE_SPLITTING_SHARED_LINE.search(line) is not None
    )

    # If we're not ignoring and we've matched an environment...
    if contains_splittable_env:
        # Return True if the comment index is None (which implies the split point must be in text),
        # otherwise compare the index of the comment with the split point
        comment_index = find_comment_index(line, pattern)
        if comment_index is None:
            return True

        match = RE_SPLITTING_SHARED_LINE_CAPTURE.search(line)
        if match and match.start(2) > comment_index:
            # If split point is past the comment index, don't split
            return False
        else:
            # Otherwise, split point is before comment and we do split
            return True
    else:
        # If ignoring or didn't match an environment, don't need a new line
        return False


def split_line(line: str, state: State, file: str, args: Args, logs: List[Log]) -> Tuple[str, str]:
    """Ensure lines are split correctly."""
    match = RE_SPLITTING_SHARED_LINE_CAPTURE.search(line)
    if not match:
        return line, ""

    prev = match.group("prev")
    rest = match.group("env")

    if args.verbosity >= 3:  # Trace level
        logs.append(
            Log(
                level="TRACE",
                file=file,
                message="Placing environment on new line.",
                linum_new=state.linum_new,
                linum_old=state.linum_old,
                line=line,
            )
        )

    return prev, rest


def set_ignore_and_report(
    line: str, temp_state: State, logs: List[Log], file: str, pattern: Pattern
) -> bool:
    """Sets the ignore and verbatim flags in the given State based on line and returns whether line should be ignored."""
    temp_state.ignore = get_ignore(line, temp_state, logs, file, True)
    temp_state.verbatim = get_verbatim(line, temp_state, logs, file, True, pattern)

    return temp_state.verbatim.visual or temp_state.ignore.visual


def clean_text(text: str, args: Args) -> str:
    """Cleans the given text by removing extra line breaks and trailing spaces."""
    # Remove extra newlines
    text = RE_NEWLINES.sub(f"{LINE_END}{LINE_END}", text)

    # Remove tabs if they shouldn't be used
    if args.tabchar != "\t":
        text = text.replace("\t", " " * args.tabsize)

    # Remove trailing spaces
    text = RE_TRAIL.sub(LINE_END, text)

    return text


def remove_trailing_spaces(text: str) -> str:
    """Remove trailing spaces from line endings."""
    return RE_TRAIL.sub(LINE_END, text)


def remove_trailing_blank_lines(text: str) -> str:
    """Remove trailing blank lines from file."""
    return text.rstrip() + LINE_END


def indents_return_to_zero(state: State) -> bool:
    """Check if indentation returns to zero at the end of the file."""
    return state.indent.actual == 0


def format_latex(text: str) -> str:
    """Format LaTeX text with default formatting options.

    This is the main API function for formatting LaTeX text.
    It uses pre-defined default values for all formatting parameters.

    Args:
        text: LaTeX text to format

    Returns:
        Formatted LaTeX text
    """
    # Use default configuration
    args = Args()
    file = "input.tex"

    # Format and return only the text
    formatted_text, _ = _format_latex(text, file, args)
    return formatted_text.strip()


def _format_latex(old_text: str, file: str, args: Args) -> Tuple[str, List[Log]]:
    """Internal function to format a LaTeX string."""
    logs = []
    logs.append(Log(level="INFO", file=file, message="Formatting started."))

    # Clean the source file
    old_text = clean_text(old_text, args)
    old_lines = list(enumerate(old_text.splitlines(), 1))

    # Initialize
    state = State()
    queue = []
    new_text = ""

    # Select the character used for indentation
    indent_char = "\t" if args.tabchar == "\t" else " "

    # Get any extra environments to be indented as lists
    lists_begin = [f"\\begin{{{l}}}" for l in args.lists]
    lists_end = [f"\\end{{{l}}}" for l in args.lists]

    while True:
        if queue:
            linum_old, line = queue.pop(0)

            # Read the patterns present on this line
            pattern = Pattern.new(line)

            # Temporary state for working on this line
            temp_state = State(
                linum_old=linum_old,
                linum_new=state.linum_new,
                ignore=Ignore(state.ignore.actual, state.ignore.visual),
                indent=Indent(state.indent.actual, state.indent.visual),
                verbatim=Verbatim(state.verbatim.actual, state.verbatim.visual),
                linum_last_zero_indent=state.linum_last_zero_indent,
            )

            # If the line should not be ignored...
            if not set_ignore_and_report(line, temp_state, logs, file, pattern):
                # Check if the line should be split because of a pattern that should begin on a new line
                if needs_split(line, pattern):
                    # Split the line into two...
                    this_line, next_line = split_line(line, temp_state, file, args, logs)
                    # ...and queue the second part for formatting
                    if next_line:
                        queue.insert(0, (linum_old, next_line))
                    line = this_line

                # Calculate the indent based on the current state and the patterns in the line
                indent = calculate_indent(
                    line, temp_state, logs, file, args, pattern, lists_begin, lists_end
                )

                indent_length = indent.visual * args.tabsize

                # Wrap the line before applying the indent, and loop back if the line needed wrapping
                if needs_wrap(line.lstrip(), indent_length, args):
                    wrapped_lines = apply_wrap(
                        line.lstrip(), indent_length, temp_state, file, args, logs, pattern
                    )
                    if wrapped_lines:
                        this_line, next_line_start, next_line = wrapped_lines
                        queue.insert(0, (linum_old, next_line_start + next_line))
                        queue.insert(0, (linum_old, this_line))
                        continue

                # Lastly, apply the indent if the line didn't need wrapping
                line = apply_indent(line, indent, args, indent_char)

            # Add line to new text
            state = temp_state
            new_text += line + LINE_END
            state.linum_new += 1
        elif old_lines:
            linum_old, line = old_lines.pop(0)
            queue.append((linum_old, line))
        else:
            break

    if not indents_return_to_zero(state):
        msg = f"Indent does not return to zero. Last non-indented line is line {state.linum_last_zero_indent}"
        logs.append(Log(level="WARN", file=file, message=msg))

    new_text = remove_trailing_spaces(new_text)
    new_text = remove_trailing_blank_lines(new_text)
    logs.append(Log(level="INFO", file=file, message="Formatting complete."))

    return new_text, logs
