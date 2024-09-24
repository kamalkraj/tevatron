import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from tevatron.retriever.arguments import DataArguments
import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """

        query_passage_target = None
        passage_query_target = None
        # all_queries = [f[0] for f in features]
        all_queries = []
        for f in features:
            if type(f[0]) == list:
                all_queries.extend(f[0])
            else:
                all_queries.append(f[0])
        all_passages = []
        for f in features:
            if type(f[1]) == list:
                all_passages.extend(f[1])
            else:
                all_passages.append(f[1])
        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=False, 
            truncation=True,
            max_length=self.data_args.passage_max_len-1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]
        
        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if self.data_args.dataset_type == "passage_multiquery":
            target = []
            query_passage_target = []
            # print(len(q_collated['input_ids']))
            # print(len(d_collated['input_ids']))
            # print(d_collated['input_ids'].shape)
            # import ipdb;ipdb.set_trace()
            no_of_queries = int(len(q_collated['input_ids'])/len(d_collated['input_ids']))
            # print(no_of_queries)
            for i in range(int(len(q_collated['input_ids'])/no_of_queries)):
                for _ in range(no_of_queries):
                    temp = [0] * len(d_collated['input_ids'])
                    temp[i] = 1
                    query_passage_target.append(i)
                    target.append(temp)
            query_passage_target = torch.tensor(query_passage_target)
            passage_query_target = torch.tensor(target).transpose(0, 1)
        return q_collated, d_collated, query_passage_target, passage_query_target


@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_texts = self.tokenizer(
            texts,
            padding=False, 
            truncation=True,
            max_length=max_length-1 if self.data_args.append_eos_token else max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_texts['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_texts['input_ids']]
        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return text_ids, collated_texts