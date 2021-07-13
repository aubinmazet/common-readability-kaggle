from pathlib import Path

import pandas as pd
import tensorflow as tf
from custom_roberta import RobertaModel
from data import CommonReadabilityDataset
from transformers import AutoTokenizer, TFAutoModel

CHECKPOINT = "/kaggle/input/huggingface-roberta/roberta-base/"

EPOCHS = 10
BATCH_SIZE = 8
MODEL_PATH = "model.h5"
BASE_PATH = Path("/kaggle/input/commonlitreadabilityprize/")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
base_model = TFAutoModel.from_pretrained(CHECKPOINT)

def get_training_dataset(tokenizer, percentage_validation: float = 0.75):
    train = pd.read_csv(BASE_PATH + "train.csv")
    train = train.sample(train.shape[0])
    train_df = train.iloc[: int(percentage_validation * train.shape[0])]
    val_df = train.iloc[int(percentage_validation * train.shape[0]) :]
    training_dataset = CommonReadabilityDataset(train_df, tokenizer).create_tf_dataset(
        batch_size=BATCH_SIZE
    )
    validation_dataset = CommonReadabilityDataset(val_df, tokenizer).create_tf_dataset(
        batch_size=BATCH_SIZE
    )
    return training_dataset, validation_dataset


def get_test_dataset(tokenizer):
    test_df = pd.read_csv(BASE_PATH + "test.csv")
    test_dataset = CommonReadabilityDataset(test_df, tokenizer).create_tf_dataset(
        labeled=False, ordered=True, batch_size=BATCH_SIZE
    )
    return test_dataset


def train(model, training_dataset, validation_dataset):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_root_mean_squared_error",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )
    history = model.fit(
        x=training_dataset,
        validation_data=validation_dataset,
        # steps_per_epoch=50,
        callbacks=[checkpoint_callback],
        epochs=EPOCHS,
        verbose=1,
    ).history
    return history
