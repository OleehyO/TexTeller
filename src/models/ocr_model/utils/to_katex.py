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
    
    if old_inst != new_inst and (old_inst + old_surr_l) in result:
        return change(result, old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r)
    else:
        return result


def find_substring_positions(string, substring):
    positions = [match.start() for match in re.finditer(re.escape(substring), string)]
    return positions


def rm_dollar_surr(content):
    pattern = re.compile(r'\\[a-zA-Z]+\$.*?\$|\$.*?\$')
    matches = pattern.findall(content)
    
    for match in matches:
        if not re.match(r'\\[a-zA-Z]+', match):
            new_match = match.strip('$')
            content = content.replace(match, ' ' + new_match + ' ')
    
    return content


def change_all(input_str, old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r):
    pos = find_substring_positions(input_str, old_inst + old_surr_l)
    res = list(input_str)
    for p in pos[::-1]:
        res[p:] = list(change(''.join(res[p:]), old_inst, new_inst, old_surr_l, old_surr_r, new_surr_l, new_surr_r))
    res = ''.join(res)
    return res


def to_katex(formula: str) -> str:
    res = formula
    # remove mbox surrounding
    res = change_all(res, r'\mbox ', r' ', r'{', r'}', r'', r'')
    res = change_all(res, r'\mbox', r' ', r'{', r'}', r'', r'')
    # remove hbox surrounding
    res = re.sub(r'\\hbox to ?-? ?\d+\.\d+(pt)?\{', r'\\hbox{', res)
    res = change_all(res, r'\hbox', r' ', r'{', r'}', r'', r' ')
    # remove raise surrounding
    res = re.sub(r'\\raise ?-? ?\d+\.\d+(pt)?', r' ', res)
    # remove makebox
    res = re.sub(r'\\makebox ?\[\d+\.\d+(pt)?\]\{', r'\\makebox{', res)
    res = change_all(res, r'\makebox', r' ', r'{', r'}', r'', r' ')
    # remove vbox surrounding, scalebox surrounding
    res = re.sub(r'\\raisebox\{-? ?\d+\.\d+(pt)?\}\{', r'\\raisebox{', res)
    res = re.sub(r'\\scalebox\{-? ?\d+\.\d+(pt)?\}\{', r'\\scalebox{', res)
    res = change_all(res, r'\scalebox', r' ', r'{', r'}', r'', r' ')
    res = change_all(res, r'\raisebox', r' ', r'{', r'}', r'', r' ')
    res = change_all(res, r'\vbox', r' ', r'{', r'}', r'', r' ')


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
        res = change_all(res, old_ins, new_ins, r'$', r'$', '{', '}')
    res = change_all(res, r'\boldmath ', r'\bm', r'{', r'}', r'{', r'}')
    res = change_all(res, r'\boldmath', r'\bm', r'{', r'}', r'{', r'}')
    res = change_all(res, r'\boldmath ', r'\bm', r'$', r'$', r'{', r'}')
    res = change_all(res, r'\boldmath', r'\bm', r'$', r'$', r'{', r'}')
    res = change_all(res, r'\scriptsize', r'\scriptsize', r'$', r'$', r'{', r'}')
    res = change_all(res, r'\emph', r'\textit', r'{', r'}', r'{', r'}')
    res = change_all(res, r'\emph ', r'\textit', r'{', r'}', r'{', r'}')
    
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
        res = change_all(res, origin_ins, origin_ins, r'{', r'}', r'', r'')

    res = re.sub(r'\\\[(.*?)\\\]', r'\1\\newline', res)

    if res.endswith(r'\newline'):
        res = res[:-8]

    # remove multiple spaces
    res = re.sub(r'(\\,){1,}', ' ', res)
    res = re.sub(r'(\\!){1,}', ' ', res)
    res = re.sub(r'(\\;){1,}', ' ', res)
    res = re.sub(r'(\\:){1,}', ' ', res)
    res = re.sub(r'\\vspace\{.*?}', '', res)

    # merge consecutive text
    def merge_texts(match):
        texts = match.group(0)
        merged_content = ''.join(re.findall(r'\\text\{([^}]*)\}', texts))
        return f'\\text{{{merged_content}}}'
    res = re.sub(r'(\\text\{[^}]*\}\s*){2,}', merge_texts, res)

    res = res.replace(r'\bf ', '')
    res = rm_dollar_surr(res)

    # remove extra spaces (keeping only one)
    res = re.sub(r' +', ' ', res)

    return res.strip()
