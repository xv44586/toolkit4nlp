# -*- coding: utf-8 -*-
# @Date    : 2020/7/17
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : tokenizers.py
import six
import unicodedata
import json
import re

_open_ = open
is_py2 = six.PY2
if not is_py2:
    basestring = str


def convert_to_str(text, encoding='utf-8', errors='ignore'):
    """字符串转换为str格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, unicode):
            text = text.encode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, str):
            text = text.decode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


class open:
    """模仿python自带的open函数，主要是为了同时兼容py2和py3
    """

    def __init__(self, name, mode='r', encoding=None, errors='ignore'):
        if is_py2:
            self.file = _open_(name, mode)
        else:
            self.file = _open_(name, mode, encoding=encoding, errors=errors)
        self.encoding = encoding
        self.errors = errors

    def __iter__(self):
        for l in self.file:
            if self.encoding:
                l = convert_to_unicode(l, self.encoding, self.errors)
            yield l

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def write(self, text):
        if self.encoding:
            text = convert_to_str(text, self.encoding, self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


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


def _cjk_punctuation():
    return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'


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

    def truncate_seq(self, first_seq, second_seq=None, maxlen=None, pop_index=-1):
        """按 maxlen 截断两个序列，策略是优先从较长的一个中 pop.(pop_index)"""
        if second_seq is None:
            second_seq = []
        while True:
            total_length = len(first_seq) + len(second_seq)
            if total_length <= maxlen:
                break
            elif len(first_seq) > len(second_seq):
                first_seq.pop(pop_index)
            else:
                second_seq.pop(pop_index)

        return first_seq, second_seq

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
            token_dict = load_vocab(token_dict)
        self._token_dict = token_dict
        self._do_lower_case = do_lower_case
        self._inv_token_dict = {v: k for k, v in token_dict.items()}
        self._vocab_size = len(token_dict)

        # update attribute of particular token
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            # custom tokens may not have all of this tokens
            try:
                token_id = token_dict[getattr(self, '_token_{}'.format(token))]
                setattr(self, '_token_{}_id'.format(token), token_id)
            except:
                pass

    def _basic_tokenize(self, text):
        text = convert_to_unicode(text)
        if self._do_lower_case:
            text = text.lower()
            """Strips accents from a piece of text."""
            text = unicodedata.normalize("NFD", text)
            text = [cat for cat in text if unicodedata.category(cat) != 'Mn']

        spaced_text = ''
        for char in text:
            if ord(char) == 0 or ord(char) == 0xfffd or is_control(char):  # 过滤非法字符
                continue
            elif is_punctuation(char) or is_chinese_char(char):  # 中文与标点前后加空格切分
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

    def token_to_id(self, token):
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, id_):
        return self._inv_token_dict.get(id_, self._token_unk)

    def _tokenize(self, text):
        """token text"""
        spaced_text = self._basic_tokenize(text)
        tokens = []
        for text in spaced_text.strip().split():
            tokens.extend(self._word_piece_tokenize(text))

        return tokens

    def tokenize(self, text, maxlen=None):
        """token text 并按bert格式拼装结果"""
        tokens = self._tokenize(text)
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is None:
            return tokens

        pop_index = int(bool(self._token_end)) + 1
        tokens, _ = self.truncate_seq(tokens, None, maxlen=maxlen, pop_index=-pop_index)
        return tokens

    def encode(self, first_text, second_text=None, maxlen=None, pattern='S*E*E'):
        """
        输出对应的token_ids 与 segment_ids
        :param first_text:
        :param second_text:
        :param maxlen:
        :param pattern: S*E*E / S*ES*E ,区别第二个seq是否有start_token
        :return:
        """
        if isinstance(first_text, basestring):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif isinstance(second_text, basestring):
            start_index = 0
            if pattern == 'S*E*E':
                start_index = int(bool(self._token_start)) + 1
            second_tokens = self.tokenize(second_text)[start_index:]
        else:
            second_tokens = second_text

        if maxlen is not None:
            self.truncate_seq(first_tokens, second_tokens, maxlen, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)
        second_token_ids = self.tokens_to_ids(second_tokens) if second_tokens else []
        second_segment_ids = [1] * len(second_token_ids) if second_token_ids else []
        return first_token_ids + second_token_ids, first_segment_ids + second_segment_ids

    def _is_special_token(self, token):
        # 是否是带有 [ ] 的特殊字符
        return bool(token) and token[0] == '[' and token[-1] == ']'

    def decode(self, ids, tokens=None):
        """转换为可读文本"""
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special_token(token)]  # 过滤特殊token
        text, flat = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and is_chinese_char(token):
                text += token
            elif len(token) == 1 and is_punctuation(token):
                text += token + ' '
            elif len(token) == 1 and is_chinese_char(token):
                text += token
            elif i > 0 and is_chinese_char(text[-1]):
                text += token
            else:
                text += ' ' + token
        # format
        text = re.sub(' +', ' ', text)  # 连续空格替换为单个
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)  # 右上单引号后接空格 去掉空格
        punctuation = _cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex  # 中文符号后的空格删除
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)  # 删除数字小数点中间空格
        return text.strip()

    def rematch(self, text, tokens):
        """原始text 与 tokenize后的token的映射关系"""
        if is_py2:
            text = unicode(text)
        if self._do_lower_case:
            text = text.lower()

        char_mapping = []
        normalized_text = ''
        for i, char in enumerate(text):
            if self._do_lower_case:
                char = unicodedata.normalize("NFD", char)
                char = ''.join([ch for ch in char if unicodedata.category(ch) != 'Mn'])

            char = ''.join([ch for ch in char if  not (ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch))])
            normalized_text += char
            char_mapping.extend([i]*len(char))

        token_mapping = []
        offset = 0
        for token in tokens:
            if self._is_special_token(token):
                token_mapping.append([])
            else:
                token = remove_remark(token)
                start = normalized_text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start: end])
                offset = end
        return token_mapping
