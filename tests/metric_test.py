import unittest

import torch
from torchmetrics.text import Perplexity

from src.metrics import AttackSuccessRate, TokenUsed


class TestMetricASR(unittest.TestCase):
    def test_metric_asr(self):
        asr_metric = AttackSuccessRate()
        asr_metric.update(True)
        asr_metric.update(False)
        asr_metric.update(True)
        res = asr_metric.compute()
        assert res == 2.0 / 3.0

    def test_metric_asr_no_update(self):
        asr_metric = AttackSuccessRate()
        res = asr_metric.compute()
        assert res == 0.0

    def test_metric_token_used(self):
        tokens_metric = TokenUsed()
        input_tokens = ["hello", "world"]
        tokens_metric.update(input_tokens)
        res = tokens_metric.compute()
        assert res == 2.0

    def test_perplexity(self):
        gen = torch.manual_seed(42)
        preds = torch.rand(2, 8, 5, generator=gen)
        target = torch.randint(5, (2, 8), generator=gen)
        target[0, 6:] = -100
        prep = Perplexity(ignore_index=-100)
        res = prep(preds, target)
        assert res != 0.0
