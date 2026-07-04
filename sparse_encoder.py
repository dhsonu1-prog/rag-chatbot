import re
import math
import zlib

class HashingSparseEncoder:
    """
    A deterministic sparse vector encoder using the Hashing Trick (TF-IDF equivalent).
    No vocabulary mapping files are required.
    """
    def __init__(self, num_features=1048576):  # 2**20 dimensions
        self.num_features = num_features

    def tokenize(self, text):
        # Match alphanumeric words of length >= 2
        return re.findall(r'\b\w{2,}\b', text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        # Calculate term frequencies
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        features = {}
        for t, count in tf.items():
            # Use zlib.adler32 for deterministic hashing across all Python runs/platforms
            idx = zlib.adler32(t.encode('utf-8')) % self.num_features
            # Logarithmic weighting to avoid dominance of high frequency terms
            val = float(math.log1p(count))
            features[idx] = features.get(idx, 0.0) + val

        # Qdrant expects sorted indices and float values
        sorted_features = sorted(features.items())
        indices = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]

        return {"indices": indices, "values": values}
