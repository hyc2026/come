import json
from scipy.stats import poisson
import random

def calcu_edit_distance(del_token, add_token):
    ''' What operation did del_token become add_token through '''

    m = len(del_token)
    n = len(add_token)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i

    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if del_token[i - 1] == add_token[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    i = len(dp) - 1
    j = len(dp[0]) - 1
    res = []
    while i > 0 or j > 0:
        a = dp[i - 1][j - 1] if i > 0 and j > 0 else float("inf")
        b = dp[i - 1][j] if i > 0 else float("inf")
        c = dp[i][j - 1] if j > 0 else float("inf")
        min_val = min([a, b, c])

        if dp[i][j] == a and a == min_val:
            i -= 1
            j -= 1
            assert del_token[i] == add_token[j]
            res.append((del_token[i], 0)) # ori
        elif a == min([a, b, c]):
            i -= 1
            j -= 1
            res.append((add_token[j], 2)) # add
            res.append((del_token[i], 1)) # del
        elif b == min([a, b, c]):
            i = i - 1
            res.append((del_token[i], 1)) # del
        else:
            j = j - 1
            res.append((add_token[j], 2)) # add
    res = res[::-1]
    res_token = [r[0] for r in res]
    res_tag =  [r[1] for r in res]
    return res_token, res_tag


def gen_edist(diff, tokenizer, max_len):
    """
    input:
    diff - split by '<nl> '
    output:
    diff_out - split by ' '
    tag - diff_out token tag
    """
    diff_out, tag = [], []
    i = 0
    while i < len(diff):
        if len(diff[i]) == 0:
            i += 1
            continue
        if diff[i][0] == '-':
            del_list, add_list = [], []
            while i < len(diff) and diff[i][0] == '-':
                diff_token = diff[i].split()[1:]
                diff_token = " ".join(diff_token)
                diff_token += " <nl>"
                output = tokenizer.encode(diff_token)
                del_list += output[1:-1]
                i += 1
                while i < len(diff) and len(diff[i]) == 0:
                    i += 1
            while i < len(diff) and diff[i][0] == '+':
                diff_token = diff[i].split()[1:]
                diff_token = " ".join(diff_token)
                diff_token += " <nl>"
                output = tokenizer.encode(diff_token)
                add_list += output[1:-1]
                i += 1
                while i < len(diff) and len(diff[i]) == 0:
                    i += 1
            if len(add_list) == 0:
                # only delete
                diff_out += del_list
                tag += [1] * len(del_list)
            else:
                res_token, res_tag = calcu_edit_distance(del_list, add_list)
                diff_out += res_token
                tag += res_tag
        elif diff[i][0] == '+':
            # only add
            diff_token = diff[i].split()[1:]
            diff_token = " ".join(diff_token)
            diff_token += " <nl>"
            output = tokenizer.encode(diff_token)
            diff_out += output[1:-1]
            tag += [2] * len(output[1:-1])
            i += 1
        else:
            # only origin
            diff_token = diff[i] + " <nl>"
            output = tokenizer.encode(diff_token)
            diff_out += output[1:-1]
            tag += [0] * len(output[1:-1])
            i += 1
        if len(diff_out) > max_len:
            break
    return diff_out, tag


def gen_pretrain(diff, tag, max_len, tokenizer):
    res_diff, res_tag = [], []
    i = 0
    while i < len(tag):
        if tag[i] != 0:
            for j in range(i, len(tag)):
                if tag[j] != tag[i]:
                    break
            # i 是第一个+，j是第一个非+
            length = poisson.rvs(mu=3, size=1)[0]
            # if i <= j - 1:
            #     # 只有一个token，50%几率mask
            #     pos = random.randint(i, j)
            #     if pos == i:
            #         res_diff.append(Constants.MSK)
            #     else:
            #         res_diff.append(diff[i])
            #     res_tag.append(tag[i])
            if j == i:
                break
            elif i == j - 1:
                # 只有一个token，50%几率mask
                pos = random.randint(i, j)
                if pos == i:
                    res_diff.append(tokenizer.mask_token_id)
                else:
                    res_diff.append(diff[i])
                res_tag.append(tag[i])
            else:
                pos = random.randint(i, j - 1)
                res_diff += diff[i: pos] + [tokenizer.mask_token_id]
                res_tag += tag[i: pos + 1]
                if j > pos + length:
                    res_diff += diff[pos + length: j]
                    res_tag += tag[pos + length: j]
            i = j
        else:
            res_diff.append(diff[i])
            res_tag.append(tag[i])
            i += 1
    if len(res_diff) > max_len:
        res_diff = res_diff[:max_len - 1] + [res_diff[-1]]
        res_tag = res_tag[:max_len - 1] + [res_tag[-1]]
    assert len(res_diff) == len(res_tag)
    return res_diff, res_tag


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    if args.data_type == 's1' or args.data_type == 's2':
        diff_out, tag = gen_edist(source_str.strip().split('<nl> '), tokenizer, args.max_source_length)
        assert len(diff_out) == len(tag)
        diff_out = [tokenizer.bos_token_id] + diff_out[:args.max_source_length - 2] + [tokenizer.eos_token_id]
        tag = [0] + tag[:args.max_source_length - 2] + [0]
        pad_len = args.max_source_length - len(diff_out)
        diff_out += [tokenizer.pad_token_id] * pad_len
        tag += [3] * pad_len
        assert diff_out.count(tokenizer.eos_token_id) == 1
        if args.data_type == 's2':
            if stage == 'test':
                target_ids = []
            else:
                target_str = example.target
                if args.add_lang_ids:
                    target_str = add_lang_by_task(example.target, args.task, args.sub_task)
                if args.task in ['defect', 'clone']:
                    if target_str == 0:
                        target_str = 'false'
                    elif target_str == 1:
                        target_str = 'true'
                    else:
                        raise NameError
                target_str = target_str.replace('</s>', '<unk>')
                target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                            truncation=True)
                assert target_ids.count(tokenizer.eos_token_id) == 1

            return InputFeatures(
                example_index,
                diff_out,
                target_ids,
                url=example.url,
                tag_ids=tag
            )
        else:
            pretrain_src, pretrain_tag = gen_pretrain(diff_out, tag, args.max_target_length, tokenizer)
            return InputFeatures(
                example_index,
                pretrain_src,
                diff_out,
                url=example.url,
                tag_ids=pretrain_tag
            )

    else:
        source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
        assert source_ids.count(tokenizer.eos_token_id) == 1
        if stage == 'test':
            target_ids = []
        else:
            target_str = example.target
            if args.add_lang_ids:
                target_str = add_lang_by_task(example.target, args.task, args.sub_task)
            if args.task in ['defect', 'clone']:
                if target_str == 0:
                    target_str = 'false'
                elif target_str == 1:
                    target_str = 'true'
                else:
                    raise NameError
            target_str = target_str.replace('</s>', '<unk>')
            target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                        truncation=True)
            assert target_ids.count(tokenizer.eos_token_id) == 1

        return InputFeatures(
            example_index,
            source_ids,
            target_ids,
            url=example.url
        )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None,
                 tag_ids=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url
        self.tag_ids = tag_ids


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_jit_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    url=js['label']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
