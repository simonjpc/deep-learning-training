# char based tokenization and not BPE (for simplicity)
class SimpleTokenizer:

    def __init__(self, text):
        self.text = text
        self.vocab = list(set(text))
        self.encoder_map = {v: i for i, v in enumerate(self.vocab)}
        self.decoder_map = {i: v for i, v in enumerate(self.vocab)}

    def get_encode_fn(self):
        return lambda sentence: [self.encoder_map[character] for character in sentence]
    def get_decode_fn(self):
        return lambda lofints: "".join([self.decoder_map[i] for i in lofints])