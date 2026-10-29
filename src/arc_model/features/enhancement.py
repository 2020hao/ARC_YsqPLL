"""Training-normal median/MAD reference and signed-log enhancement."""

import numpy as np


class FeatureTransform:
    def fit(self, x, y, mode):
        self.mode = mode
        normal = x[np.asarray(y) == 0]
        self.med = np.median(normal, axis=0)
        self.mad = np.maximum(1.4826 * np.median(np.abs(normal - self.med), axis=0), 0.02)
        z = self.enhance(x)
        self.mean = z.mean(0)
        self.std = np.maximum(z.std(0), 0.05)
        return self

    def enhance(self, x):
        if self.mode == "raw":
            return x
        z = (x - self.med) / self.mad
        return np.sign(z) * np.log1p(np.abs(z))

    def transform(self, x):
        return ((self.enhance(x) - self.mean) / self.std).astype("float32")

    def state(self):
        return dict(mode=self.mode, med=self.med, mad=self.mad, mean=self.mean, std=self.std)
