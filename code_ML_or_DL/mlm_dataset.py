"""实现bert 的masked language model的dataset，15%mask，且这15%中，80%进行mask，10%不变，10%随机替换
参考fairseq，位置：fairseq/fairseq/data/mask_tokens_dataset.py
"""
import torch
import numpy as np


class MLMDataset(torch.utils.data.Dataset):
    # bert 的Masked Language Model的dataset
    def __init__(
        self,
        dataset,
        mask_idx,
        pad_idx,
        vocab_size,
        special_token_idx=[0, 1],  # mask_idx, pad_idx等特殊token的idx，在随机替换时不考虑这些特殊token
        mask_prob: float = 0.15,
        keep_unchange_prob: float = 0.1,  # 保持不变
        random_replace_prob: float = 0.1,  # 随机替换
    ):
        super().__init__()
        self.dataset = dataset
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.mask_prob = mask_prob
        self.keep_unchange_prob = keep_unchange_prob
        self.random_replace_prob = random_replace_prob
        self.vocab_size = vocab_size
        if self.random_replace_prob > 0.0:
            rand_replace_weights = np.ones(self.vocab_size)
            rand_replace_weights[special_token_idx] = 0
            self.rand_replace_weights = (
                rand_replace_weights / rand_replace_weights.sum()
            )  # 用于随机替换时候的采样

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        length = len(item)
        mask = np.full(length, False)  # 其中length为这个样本的长度
        num_mask = int(self.mask_prob * length + np.random.rand())
        mask_idx = np.random.choice(length, num_mask, replace=False)
        mask[mask_idx] = True

        # target for masked LM training，其中算loss的时候，遇到self.pad_idx直接忽略
        target = np.full(len(mask), self.pad_idx)
        #  下面的#1和#2等价
        # target[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]  #1
        target[mask_idx] = item[torch.from_numpy(mask_idx)]  # 2
        target = torch.from_numpy(target)

        # sources for masked LM training，
        rand_or_keep_prob = (
            self.random_replace_prob + self.keep_unchange_prob
        )  # random_token_prob 和 keep_unchange_prob 这两个概率值均为0.1，加起来0.1+0.1=0.2
        rand_or_keep = mask & (np.random.rand(length) < rand_or_keep_prob)

        # rand_or_keep 部分,分别有50%的概率是随机替换(rand)，50%是不变(keep)
        keep_prob = self.keep_unchange_prob / rand_or_keep_prob
        decision = np.random.rand(length) < keep_prob  # keep_prob值为50%
        keep_mask = rand_or_keep & decision
        rand_mask = rand_or_keep & (~decision)

        # 对于keep_unchange部分， 其src是不变的
        mask = mask ^ keep_mask

        source = np.copy(item)
        source[mask] = self.mask_idx  # mask idx

        # random_token 部分随机选择
        num_rand = rand_mask.sum()
        source[rand_mask] = np.random.choice(
            self.vocab_size, num_rand, p=self.rand_replace_weights
        )  # rand replace idx
        source = torch.from_numpy(source)
        return source, target


if __name__ == "__main__":
    # test
    sentences = [[2] * 20 for _ in range(3)]
    dataset = torch.tensor(sentences)
    mask_idx, pad_idx, vocab_size = 0, 1, 200
    masked_language_model_dataset = MLMDataset(
        dataset, mask_idx, pad_idx, vocab_size
    )
    for i, sample in enumerate(masked_language_model_dataset):
        source, target = sample
        print(f"source={source}")
        print(f"target={target}")
        break
