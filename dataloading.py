import chardet
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


def load_data(data_path):
    dataset = load_files(data_path)
    for i in range(len(dataset.data)):
        if chardet.detect(dataset.data[i]) != "utf-8":
            dataset.data[i] = dataset.data[i].decode(chardet.detect(dataset.data[i])['encoding']).encode('utf8')
    dataset.data = [i.replace(b"\n", b" ") for i in dataset.data]
    return pd.DataFrame({'text': dataset.data, 'target': dataset.target})


def plot_class_distribution(df, title):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title(title)
    plt.show()


class NewsDataset(Dataset):
    def __init__(self, df, max_token_len, tokenizer):
        df.reset_index(drop=True, inplace=True)
        self.text = df.text
        self.targets = df.target
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
        )
        token_ids = inputs['input_ids']
        token_mask = inputs['attention_mask']
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'token_mask': torch.tensor(token_mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


class NewsDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path,
                 batch_size,
                 max_token_len=512,
                 dataloader_pin_memory=False,
                 dataloader_num_workers=0,
                 ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.dataloader_pin_memory = dataloader_pin_memory,
        self.dataloader_num_workers = dataloader_num_workers
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        self.class_labels = ['athletics', 'cricket', 'football', 'rugby', 'tennis']

    def setup(self, stage=None):
        print("\nLoad Dataset...")
        df = load_data(f"{self.data_path}")

        train_df, test_val_df = train_test_split(df, train_size=0.7, random_state=1234)
        val_df, test_df = train_test_split(test_val_df, train_size=0.5, random_state=1234)

        if stage == 'fit':
            plot_class_distribution(df, 'Class Distribution - Entire Dataset')
            plot_class_distribution(train_df, 'Class Distribution - Training Set')
            plot_class_distribution(val_df, 'Class Distribution - Validation Set')
            plot_class_distribution(test_df, 'Class Distribution - Test Set')

        print(f"# of Train Dataset : {len(train_df)}")
        print(f"# of Test Dataset : {len(test_df)}")
        print(f"# of Valid Dataset : {len(val_df)}\n")

        self.train_dataset = NewsDataset(train_df, self.max_token_len, self.tokenizer)
        self.test_dataset = NewsDataset(test_df, self.max_token_len, self.tokenizer)
        self.valid_dataset = NewsDataset(val_df, self.max_token_len, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,
                          pin_memory=self.dataloader_pin_memory, num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn,
                          pin_memory=self.dataloader_pin_memory,
                          num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn,
                          pin_memory=self.dataloader_pin_memory,
                          num_workers=self.dataloader_num_workers)

    @staticmethod
    def collate_fn(data):
        return {"token_ids": torch.vstack([d["token_ids"] for d in data if d != -1]),
                "token_mask": torch.vstack([d["token_mask"] for d in data if d != -1]),
                "targets": torch.tensor([d["targets"] for d in data], dtype=torch.long)}
