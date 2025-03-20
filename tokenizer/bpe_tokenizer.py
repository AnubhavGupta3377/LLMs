'''
References:
1. Andrej Karpathy's video: https://www.youtube.com/watch?v=zduSFxRajkE&t=5128s&ab_channel=AndrejKarpathy
'''

class BPE_Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    # Get the counts of each id pair
    def _get_pair_counts(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    # Merge the most frequent pair
    def _merge_pair(self, ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size=500):
        raw_bytes = text.encode("utf-8")
        token_ids = list(raw_bytes)

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        num_merges = vocab_size - 256
        for i in range(num_merges):
            new_id = 256 + i
            pair_counts = self._get_pair_counts(token_ids)
            # Get the most frequent pair
            pair = max(pair_counts, key=lambda x: x[1])

            # Merge the pair
            print(f"Merging pair {pair} into new token id {new_id}")
            token_ids = self._merge_pair(token_ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if len(token_ids) == 1:
                break
    
    def encode(self, text):
        raw_bytes = text.encode("utf-8")
        token_ids = list(raw_bytes)
        while True:
            pair_counts = self._get_pair_counts(token_ids)
            merge_pair = min(pair_counts, key=lambda pair: self.merges.get(pair, float("inf")))
            if merge_pair not in self.merges:
                break
            token_ids = self._merge_pair(token_ids, merge_pair, self.merges[merge_pair])
        return token_ids
    
    # Given a list of token ids, decode it into text
    def decode(self, token_ids):
        token_bytes = b"".join([self.vocab[token_id] for token_id in token_ids])
        text = token_bytes.decode("utf-8", errors="replace")
        return text
                

text = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes"""
tokenizer = BPE_Tokenizer()
tokenizer.train(text, vocab_size=280)
encoded = tokenizer.encode(text)
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
print(text == decoded)
