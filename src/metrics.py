import torch
from torchmetrics import Metric


# For Perplexity metric we will use the torchmetrics library
# from torchmetrics.text import Perplexity

class AttackSuccessRate(Metric):
    higher_is_better = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("attack_success", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, success: bool):
        if success:
            self.attack_success += 1
        self.total += 1

    def compute(self):
        if self.total == 0 or self.attack_success == 0:
            return torch.tensor(0.0)
        return self.attack_success.float() / self.total


class TokenUsed(Metric):
    higher_is_better = False

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.add_state("tokens_used", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tokens: list[str]):
        self.tokens_used += len(tokens)

    def compute(self):
        return self.tokens_used.float()