from datasets import load_dataset
from torch.utils.data import Dataset


class wikihow(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        type_path,
        num_samples,
        input_length,
        output_length,
        print_text=False,
    ):
        self.dataset = load_dataset("FiscalNote/billsum", split=type_path)

        # if type_path=="train":
        #     print(self.dataset.column_names)
        #     print("Size of train dataset: ", self.dataset['train'].shape)
        #     print("Size of Validation dataset: ", self.dataset['validation'].shape)

        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        text = text.replace("Example of text:", "")
        text = text.replace("Example of Summary:", "")
        text = text.replace("\n", "")
        text = text.replace("``", "")
        text = text.replace('"', "")

        return text

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["text"]))
        #         input_ = self.clean_text(example_batch['text']) + " </s>"
        #         target_ = self.clean_text(example_batch['headline']) + " </s>"

        input_ = self.clean_text(example_batch["text"])
        target_ = self.clean_text(example_batch["summary"])

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }


# def get_dataset(tokenizer, type_path, num_samples, args):
#       return wikihow(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=max_input_length,
#                         output_length=max_output_length)