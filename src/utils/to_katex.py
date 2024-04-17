import numpy as np
import re

def to_katex(formula: str) -> str:
    res = formula
    res = re.sub(r'\\mbox\{([^}]*)\}', r'\1', res)
    res = re.sub(r'boldmath\$(.*?)\$', r'bm{\1}', res)
    res = re.sub(r'\\\[(.*?)\\\]', r'\1\\newline', res) 

    pattern = r'(\\(?:left|middle|right|big|Big|bigg|Bigg|bigl|Bigl|biggl|Biggl|bigm|Bigm|biggm|Biggm|bigr|Bigr|biggr|Biggr))\{([^}]*)\}'
    replacement = r'\1\2'
    res = re.sub(pattern, replacement, res)
    if res.endswith(r'\newline'):
        res = res[:-8]
    return res
