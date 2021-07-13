import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer


# Datasets utility functions
def custom_standardization(text):
    text = text.lower()  # if encoder is uncased
    text = text.strip()
    return text


class CommonReadabilityDataset:
    MAX_LENGTH = 256

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def create_tf_dataset(
        self,
        labeled: bool = True,
        repeated: bool = False,
        ordered: bool = False,
        batch_size: int = 8,
    ):
        df = self.df.copy()
        text = [custom_standardization(text) for text in df["excerpt"]]
        # Tokenize inputs
        tokenized_inputs = self.tokenizer(
            text,
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
        )
        if labeled:
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "input_ids": tokenized_inputs["input_ids"],
                        "attention_mask": tokenized_inputs["attention_mask"],
                    },
                    (df["target"], df["standard_error"]),
                )
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                {
                    "input_ids": tokenized_inputs["input_ids"],
                    "attention_mask": tokenized_inputs["attention_mask"],
                }
            )

        if repeated:
            dataset = dataset.repeat()
        if not ordered:
            dataset = dataset.shuffle(1024)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
