"""Tweaked AllenNLP dataset reader."""
import logging
import re
from random import random
from typing import Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from utils.helpers import SEQ_DELIMETERS, START_TOKEN
from utils.preprocess_data import convert_tagged_line, align_sequences

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2labels_datareader")
class Seq2LabelsDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    delimiters: ``dict``
        The dcitionary with all delimeters.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    max_len: if set than will truncate long sentences
    """
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 tag_strategy: str = "keep_one",
                 first_type: str = "random",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._first_type = first_type
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob

    def mask_tags(self, tags, deal=None):
        new_tags = tags.copy()
        mask_index = [1.]*len(new_tags)
        for index in range(len(tags)):
            if deal and deal in tags[index]:
                continue
            if tags[index] != "$KEEP" and random() < 0.5:
                mask_index[index]=0.
                new_tags[index] = "$KEEP"
        return new_tags, mask_index


    def get_refer_by_tokens_and_tags(self, tokens, tags):
        if len(tokens) != len(tags):
            raise Exception("Mismatch for tokens and tags!")
        aligned_sent = self._delimeters["tokens"].join(
            [tokens[i] + self._delimeters["labels"] + tags[i] for i in range(len(tokens))])
        refer_sent = convert_tagged_line(aligned_sent)
        return refer_sent

    def convert_line_to_tokens_and_tags(self, line):
        tokens_and_tags = [pair.rsplit(self._delimeters['labels'], 1)
                           for pair in line.split(self._delimeters['tokens'])]
        try:
            tokens = [Token(token) for token, tag in tokens_and_tags]
            tags = [tag for token, tag in tokens_and_tags]
        except ValueError:
            tokens = [Token(token[0]) for token in tokens_and_tags]
            tags = None

        if tokens and tokens[0] != Token(START_TOKEN):
            tokens = [Token(START_TOKEN)] + tokens
        return tokens, tags

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                # skip blank and broken lines
                if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                    continue

                
                tokens, tags = self.convert_line_to_tokens_and_tags(line)
                words = [x.text for x in tokens]
                edit_count = (len(tags) - tags.count('$KEEP'))               
                mask_tags, better_tokens, gold_tags, mask = None, None, None, None
                if self._first_type =='random':
                    deal = None
                else:
                    deal = f'${self._first_type.upper()}'

                if edit_count >= 2:
                    mask_tags, mask = self.mask_tags(tags, deal)
                    mask_count = len(mask)-mask.count(1.)
                    while (mask_count == 0  or  mask_count == edit_count):
                        mask_tags, mask = self.mask_tags(tags, deal)
                        mask_count = len(mask)-mask.count(1.)
                        if mask_count == 0:
                            break
                    '''                            
                    if mask_count!=0:
                        better_sent = self.get_refer_by_tokens_and_tags(words, mask_tags)
                        gold_sent = self.get_refer_by_tokens_and_tags(words, tags)

                        new_line = align_sequences(better_sent, gold_sent)
                        better_tokens, gold_tags = self.convert_line_to_tokens_and_tags(new_line)
                        
                    else:
                        mask_tags, better_tokens, gold_tags, mask = None, None, None, None
                    '''
                    better_sent = self.get_refer_by_tokens_and_tags(words, mask_tags)
                    gold_sent = self.get_refer_by_tokens_and_tags(words, tags)

                    new_line = align_sequences(better_sent, gold_sent)
                    better_tokens, gold_tags = self.convert_line_to_tokens_and_tags(new_line)
                    
                if self._max_len is not None:
                    tokens = tokens[:self._max_len]
                    tags = None if tags is None else tags[:self._max_len]
                    mask_tags = None if mask_tags is None else mask_tags[:self._max_len]
                    mask = None if mask is None else mask[:self._max_len]
                    better_tokens = None if better_tokens is None else better_tokens[:self._max_len]
                    gold_tags = None if gold_tags is None else gold_tags[:self._max_len]
                instance = self.text_to_instance(tokens, better_tokens, tags, mask_tags, gold_tags, mask, words)
                if instance:
                    yield instance


    def extract_tags(self, tags: List[str]):
        op_del = self._delimeters['operations']

        labels = [x.split(op_del) for x in tags]

        comlex_flag_dict = {}
        # get flags
        for i in range(5):
            idx = i + 1
            comlex_flag_dict[idx] = sum([len(x) > idx for x in labels])

        if self._tag_strategy == "keep_one":
            # get only first candidates for r_tags in right and the last for left
            labels = [x[0] for x in labels]
        elif self._tag_strategy == "merge_all":
            # consider phrases as a words
            pass
        else:
            raise Exception("Incorrect tag strategy")

        detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels]
        return labels, detect_tags, comlex_flag_dict

    def text_to_instance(self, tokens: List[Token], better_tokens: List[Token],
                         tags: List[str] = None,mask_tags: List[str] = None, gold_tags: List[str] = None,
                         mask: List[float] = None, words: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        better_sequence = TextField([] if better_tokens is None else better_tokens,self._token_indexers)

        fields["tokens"] = sequence
        fields["better_tokens"] = better_sequence

        fields["metadata"] = MetadataField({"words": words, "mask": [] if mask is None else mask})
        fields["better_metadata"] = MetadataField({"words":['$START'] if better_tokens is None else [x.text for x in better_tokens]})
        if tags is not None:
            labels, detect_tags, complex_flag_dict = self.extract_tags(tags)
            mask_labels, mask_detect_tags, mask_complex_flag_dict = self.extract_tags(tags if mask_tags is None else mask_tags)
            gold_labels, gold_detect_tags, gold_complex_flag_dict = self.extract_tags([] if gold_tags is None else gold_tags)

            if self._skip_complex and complex_flag_dict[self._skip_complex] > 0:
                return None

            rnd = random()
            # skip TN
            if self._skip_correct and all(x == "CORRECT" for x in detect_tags):
                if rnd > self._tn_prob:
                    return None
            # skip TP
            else:
                if rnd > self._tp_prob:
                    return None

            fields["labels"] = SequenceLabelField(labels, sequence,
                                                  label_namespace="labels")
            fields["d_tags"] = SequenceLabelField(detect_tags, sequence,
                                                  label_namespace="d_tags")
            fields["mask_labels"] = SequenceLabelField(mask_labels, sequence,
                                                  label_namespace="labels")
            fields["mask_d_tags"] = SequenceLabelField(mask_detect_tags, sequence,
                                                  label_namespace="d_tags")
            fields["gold_labels"] = SequenceLabelField(gold_labels, better_sequence,
                                                  label_namespace="labels")
            fields["gold_d_tags"] = SequenceLabelField(gold_detect_tags, better_sequence,
                                                       label_namespace="d_tags")
        return Instance(fields)
