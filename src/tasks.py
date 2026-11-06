from .templates import SQuADTemplate, SST2Template
from .utils import temp_seed
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Union
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_task(task_name):
    return getattr(sys.modules[__name__], f"{task_name}Dataset")()


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    train_sep = "\n\n"
    generation = False

    def sample_train_set(self, num_train, num_dev, seed):
        if num_dev is None:
            return self.sample_subset(data_split="train", seed=seed, num=num_train)
        samples = self.sample_subset(data_split="train", seed=seed, num=num_train + num_dev)
        logger.info(f"Sample train set {len(samples)}/{len(self.samples['train'])}")
        logger.info(f"... including dev set {num_dev} samples")
        return samples

    def sample_subset(self, data_split="train", seed=0, num=100):
        with temp_seed(seed):
            samples = self.samples[data_split]
            index = np.random.permutation(len(samples)).tolist()[:num]
            return [samples[i] for i in index]


class SST2Dataset(Dataset):
    def __init__(self):
        self.load_dataset()

    def load_dataset(self):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self):
        return SST2Template()


class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self):
        self.load_dataset()

    def load_dataset(self):
        # plain_text (SQuAD v1.1) is named because a cache that also holds the
        # `default` config makes an unnamed load ambiguous.
        dataset = load_dataset("squad", "plain_text")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    def get_template(self):
        return SQuADTemplate()
