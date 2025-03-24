from datasets import load_dataset

# dataset
def get_dataset():
    dataset = load_dataset("alpindale/light-novels")
    text_dataset = "\n".join(dataset["train"]["text"])
    return text_dataset