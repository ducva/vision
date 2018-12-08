from fastai.text.transform import Tokenizer

from vision.imports.core import *

__all__ = ["TextTokenizer"]

BOS, FLD, UNK, PAD = 'xxbos', 'xxfld', 'xxunk', 'xxpad'
TK_MAJ, TK_UP, TK_REP, TK_WREP = 'xxmaj', 'xxup', 'xxrep', 'xxwrep'
default_spec_tok = [BOS, FLD, UNK, PAD]


def spec_add_spaces(t: str) -> str:
    """Add spaces around / and # in `t`."""
    return re.sub(r'([/#])', r' \1 ', t)


def rm_useless_spaces(t: str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(' {2,}', ' ', t)


def replace_rep(t: str) -> str:
    "Replace repetitions at the character level in `t`."

    def _replace_rep(m: Collection[str]) -> str:
        c, cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)


def replace_wrep(t: str) -> str:
    "Replace word repetitions in `t`."

    def _replace_wrep(m: Collection[str]) -> str:
        c, cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(_replace_wrep, t)


def fix_html(x: str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', UNK).replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def replace_all_caps(x: Collection[str]) -> Collection[str]:
    "Add `TK_UP` for words in all caps in `x`."
    res = []
    for t in x:
        if t.isupper() and len(t) > 1: res.append(TK_UP)
        res.append(t)
    return res


def deal_caps(x: Collection[str]) -> Collection[str]:
    "Replace all words in `x` by their lower version and add `TK_MAJ`."
    res = []
    for t in x:
        if t[0].isupper() and t[1:].islower(): res.append(TK_MAJ)
        res.append(t.lower())
    return res


class TextTokenizer:
    def __init__(self, lang: str = 'en'):
        default_post_rules = [replace_all_caps, deal_caps]
        default_pre_rules = [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces]
        self.tok = Tokenizer(lang=lang, pre_rules=default_pre_rules, post_rules=default_post_rules)

    def process_all(self, texts: list):
        return self.tok.process_all(texts)
