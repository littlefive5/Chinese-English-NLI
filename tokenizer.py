import pkuseg
import nltk.tokenize as tk
import collections
seg = pkuseg.pkuseg()

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, vocab_file,languages, do_lower_case=True):
        """ Constructs a BasicTokenizer.
        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        self.do_lower_case = do_lower_case
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.languages=languages
        if self.languages == 'Chinese':
            self.seg = pkuseg.pkuseg()
        elif self.languages == 'English':
            self.seg = tk
    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
            Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.
        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        if self.languages == 'Chinese':
            output_tokens = self.seg.cut(text)
        else:
            if self.do_lower_case:
                text = text.lower()
            output_tokens = self.seg.word_tokenize(text)
        return output_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get('[UNK]'))
    def convert_tokens_to_ids(self, tokens):
        """ Converts a token string (or a sequence of tokens) in a single integer id
            (or a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    @property
    def vocab_size(self):
        return len(self.vocab)