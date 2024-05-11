import re


def change(input_str, old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r):
    result = ""
    i = 0
    n = len(input_str)
    
    while i < n:
        if input_str[i:i+len(old_inst)] == old_inst:
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
                if input_str[j] == '\\' and not escaped:
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
                inner_content = input_str[start + 1:j]
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
    
    if old_inst != new_inst and old_inst in result:
        return change(result, old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r)
    else:
        return result


def to_katex(formula: str) -> str:
    res = formula
    res = change(res, r'\mbox ', r'', r'{', r'}', r'', r'')
    res = change(res, r'\mbox', r'', r'{', r'}', r'', r'')

    origin_instructions = [
        r'\Huge',
        r'\huge',
        r'\LARGE',
        r'\Large',
        r'\large',
        r'\normalsize',
        r'\small',
        r'\footnotesize',
        r'\tiny'
    ]
    for (old_ins, new_ins) in zip(origin_instructions, origin_instructions):
        res = change(res, old_ins, new_ins, r'$', r'$', '{', '}')
    res = change(res, r'\boldmath ', r'\bm', r'$', r'$', r'{', r'}')
    res = change(res, r'\boldmath', r'\bm', r'$', r'$', r'{', r'}')
    res = change(res, r'\scriptsize', r'\scriptsize', r'$', r'$', r'{', r'}')
    
    origin_instructions = [
        r'\left',
        r'\middle',
        r'\right',
        r'\big',
        r'\Big',
        r'\bigg',
        r'\Bigg',
        r'\bigl',
        r'\Bigl',
        r'\biggl',
        r'\Biggl',
        r'\bigm',
        r'\Bigm',
        r'\biggm',
        r'\Biggm',
        r'\bigr',
        r'\Bigr',
        r'\biggr',
        r'\Biggr'
    ]
    for origin_ins in origin_instructions:
        res = change(res, origin_ins, origin_ins, r'{', r'}', r'', r'')

    res = re.sub(r'\\\[(.*?)\\\]', r'\1\\newline', res)

    if res.endswith(r'\newline'):
        res = res[:-8]
    return res
