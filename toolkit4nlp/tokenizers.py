# -*- coding: utf-8 -*-
# @Date    : 2020/7/17
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : tokenizers.py
import six
import unicodedata
import json

is_py2 = six.PY2
if not is_py2:
    basestring = str


def load_vocab(vocab_path, encoding='utf8', simplified=False, startswith=None):
    """
    从字典文件中读入字典
    """
    token_dict = {}
    with open(vocab_path, encoding=encoding) as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    if not simplified:
        return token_dict

    new_token_dict = {}
    keep_tokens = []
    if startswith is not None:
        for s in startswith:
            new_token_dict[s] = len(new_token_dict)
            keep_tokens.append(token_dict[s])

    for t, _ in sorted(token_dict.items(), key=lambda kv: kv[1]):
        if t not in new_token_dict:
            keep = True
            if len(t) > 1:
                for c in remove_remark(t):
                    if is_punctuation(c) or is_chinese_char(c):
                        keep = False
                        break
            if keep:
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])

    return new_token_dict, keep_tokens


def save_vocab(save_path, token_dict, encoding='utf8'):
    with open(save_path, 'w', encoding=encoding) as fout:
        for k, _ in sorted(token_dict.items(), key=lambda kv: kv[1]):
            fout.write(k + '\n')


def remove_remark(token):
    # remove any '##' remark at the start of the token
    if token[:2] == '##':
        return token[2:]
    return token


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(cp)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


class BasicTokenizer(object):
    def __init__(self, start='[CLS]', end='[SEP]'):
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = start
        self._token_end = end

    def tokenize(self, text):
        raise NotImplementedError

    def encode(self, first_text, second_text=None, maxlen=None):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError

    def truncate_seq(self, first_seq, second_seq=None, maxlen=None):
        raise NotImplementedError

    def token_to_id(self, token):
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        return [self.token_to_id(token) for token in tokens]

    def id_to_token(self, id_):
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        return [self.id_to_token(id_) for id_ in ids]


class Tokenizer(BasicTokenizer):
    """Bert tokenizer"""

    def __init__(self, token_dict, do_lower_case=False, *args, **kwargs):
        super(Tokenizer, self).__init__(*args, **kwargs)
        if isinstance(token_dict, six.string_types):
            token_dict =load_vocab(token_dict)
        self.token_dict = token_dict
        self.do_lower_case = do_lower_case
        self.inv_token_dict = {v: k for k, v in token_dict.items()}

        # update attribute of particular token
        for token in ['pad','unk','mask', 'start', 'end']:
            token_id = token_dict[getattr(self, '_token_{}'.format(token))]
            setattr(self, '_token_{}'.format(token), token_id)

    def _basic_tokenize(self, text):
        text = convert_to_unicode(text)
        if self.do_lower_case:
            text = text.lower()
            """Strips accents from a piece of text."""
            text = unicodedata.normalize("NFD", text)
            text = [cat for cat in text if unicodedata.category(cat) != 'Mn']

        spaced_text = ''
        for char in text:
            if ord(char) == 0 or ord(char) == 0xfffd or is_control(char):  # 过滤非法字符
                continue
            elif is_punctuation(char) or is_chinese_char(char):  #中文与标点前后加空格切分
                spaced_text += ' ' + char + ' '
            elif is_whitespace(char):  # 空格符直接加一个 ' '
                spaced_text += ' '
            else:
                spaced_text += char
        return spaced_text

    def _word_piece_tokenize(self, word):
        # word piece tokenize
        if word in self._token_dict:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens

    def tokenize(self, text, maxlen=None):
        spaced_text = self._basic_tokenize(text)
        tokens = []
        for text in spaced_text.strip().split():
            tokens.extend(self._word_piece_tokenize(text))

        return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")