from transformers import (
    T5Model,
    T5Tokenizer
)
from torch.utils.data import DataLoader, Dataset

import numpy as np
import polars as pl
import time
import torch


class TextClassDataset(Dataset):
    def __init__(self, x: np.array):
        self.data = [x[i] for i in range(len(x))]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index]


class EmbeddingModel:
    def __init__(self, max_seq_length: int):
        # The embedding model to be used
        self.__model: T5Model = T5Model.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

        # The model tokenizer
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            "unicamp-dl/ptt5-base-portuguese-vocab", do_lower_case=True, legacy=False
        )

        # The device to run the model
        self.device = "cuda"

        # The tokenizer parameters
        self.param_tk = {
            "return_tensors": "pt",
            "return_attention_mask": False,
            "padding": "max_length",
            "max_length": max_seq_length,
            "add_special_tokens": False,
            "truncation": True
        }

    def __create_data_loader(
        self, x: pl.Series, batch_size: int, shuffle: bool
    ) -> DataLoader:
        """
        Creates the dataloader to be used by the model
        """
        x_sampled = np.array(x.fill_null(""), dtype=object)
        dataloader_set = TextClassDataset(x_sampled)
        loader = DataLoader(dataloader_set, batch_size=batch_size, shuffle=shuffle)
        return loader

    def __get_data_loader(
        self, dataset: pl.Series, batch_size: int
    ) -> DataLoader:

        return self.__create_data_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    def __averagePooling(self, embeddings: np.ndarray) -> np.ndarray:
        return torch.mean(embeddings, dtype=torch.float64)

    def __generate_embeddings(self, data_loader: DataLoader) -> pl.Series:
        outputs = list()
        steps = len(data_loader)
        with torch.no_grad():
            for i, x_train_bt in enumerate(data_loader):
                print(f"Embeddings for {i} of {steps} data loaders")
                
                # Obtain the data from the batch
                start_time = time.perf_counter()
                x_train_bt = list(x_train_bt)
                end_time = time.perf_counter()
                print(f"x_train_bt conversion to list: {end_time-start_time}")

                # Move the data to the device to train the model
                start_time = time.perf_counter()
                input = self.tokenizer(x_train_bt, **self.param_tk).to(self.device)
                end_time = time.perf_counter()
                print(f"move input to gpu: {end_time-start_time}")

                # Feedforward prediction
                start_time = time.perf_counter()
                model_output = self.__model.encoder.embed_tokens(input.input_ids)
                end_time = time.perf_counter()
                print(f"calculate embeddings: {end_time-start_time}")

                start_time = time.perf_counter()
                outputs.extend([self.__averagePooling(embedding) for embedding in model_output])
                end_time = time.perf_counter()
                print(f"add embeddings to list: {end_time-start_time}")

        return pl.Series('embeddings', outputs)

    def get_embeddings(
        self,
        dataset: pl.Series,
        batch_size: int,
    ) -> pl.Series:
        # Loads the model to the device
        self.__model = self.__model.to(self.device)

        # Divides the dataset in training and validation loaders
        print("Creating data loaders")
        data_loader = self.__get_data_loader(dataset, batch_size)

        start_time = time.perf_counter()

        # Train the model
        print("Generating embeddings")
        embeddings = self.__generate_embeddings(data_loader)
        # print(type(embeddings), len(embeddings))
        # print(embeddings)

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Process Time (sec): {duration}")
        print(len(embeddings))

        return embeddings