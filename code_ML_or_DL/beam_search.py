import torch
from torch import nn


# 实现beam search，参考：https://blog.csdn.net/qq_27590277/article/details/107853325?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-107853325-blog-121119550.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-107853325-blog-121119550.pc_relevant_default&utm_relevant_index=2


def beam_search(
    decoder: nn.Module,
    hidden: torch.Tensor,
    beam_size: int = 3,
    bos_token_idx: int = 0,
    eos_token_idx: int = 2,
    max_length: int = 20,
    vocab_size: int = 600,
):
    # decoder就是我们的模型，比如lstm，
    # hidden.shape = (beam_size, hidden_size)

    prev_words = torch.full(
        (beam_size, 1), bos_token_idx, dtype=torch.long
    )  # (beam_size, 1)
    # 此时输出序列中只有bos token
    seqs = prev_words  # (beam_size, 1)
    # 初始化scores向量为0
    lprob_sum: torch.Tensor = torch.zeros(
        beam_size
    )  # 某个生成句子，累积的log分数，shape=（beam_size,）
    complete_seqs = []
    complete_seqs_scores = []
    step = 1
    while True:
        logits, hidden = decoder(
            prev_words, hidden
        )  # logits: (beam_size, seq_len, vocab_size)，其实logits就是logits
        logits = torch.log_softmax(logits, dim=-1)  # 我们的概率采用的是log形式
        next_token_logits: torch.Tensor = logits[:, -1, :]  # 取最后一个时刻的输出，shape=(beam_size, vocab_size)
        # 累加lprob_sum的时候，注意lprob_sum此时shape = （beam_size,），shape转成（beam_size, 1）才可以与next_token_logits相加
        lprob_sum = next_token_logits + lprob_sum.unsqueeze(
            dim=-1
        )  # 累加lprob_sum，shape=(beam_size, vocab_size)
        if step == 1:
            # 因为最开始解码的时候只有一个结点<bos>,所以只需要取其中一个结点计算topk
            topk_lprob_sum, top_k_words = lprob_sum[0].topk(  # 累积分数的topk
                beam_size, dim=0, largest=True, sorted=True
            )
        else:
            # 此时要先展开再计算topk，如上图所示。
            # topk_lprob_sum: (beam_size,) top_k_words: (beam_size,)  # 累积分数
            topk_lprob_sum, top_k_words = lprob_sum.view(-1).topk(
                beam_size, 0
            )  # 累积分数的topk
        beam_id = top_k_words // vocab_size  # (beam_size)
        token_id = top_k_words % vocab_size  # (beam_size)

        # 更新生成的句子seqs，注意要reorder再拼接
        seqs = torch.cat(
            [seqs[beam_id], token_id.unsqueeze(1)], dim=1
        )  # 2 (k, step) ==> (k, step+1)

        # 当前输出的单词不是eos的有哪些(输出其在next_wod_idx中的位置, 实际是beam_id)
        incomplete_idx = [
            idx
            for idx, next_word in enumerate(token_id)
            if next_word != eos_token_idx
            # if next_word < 300
        ]
        # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
        complete_idx = list(set(range(len(token_id))) - set(incomplete_idx))
        lprob_sum = topk_lprob_sum[beam_id]  # 3 对累积分数进行reorder，最后shape=(beam_size,)

        if len(complete_idx) > 0:
            complete_seqs.extend(seqs[complete_idx].tolist())  # 加入句子
            complete_seqs_scores.extend(
                lprob_sum[complete_idx]  # 累积分数
            )  # 加入句子对应的累加log_prob
        # 减掉已经完成的句子的数量，更新beam_size, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
        beam_size -= len(complete_idx)

        if beam_size == 0:  # 完成
            break

        # 更新下一次迭代数据, 仅专注于那些还没完成的句子，主要是完成重排序（reorder）
        # 有些代码地方使用torch.index_select(dim=,index=)进行reorder
        seqs = seqs[
            incomplete_idx
        ]  # 这里没有reorder，是因为#2那里已经完成了reorder，#2 那里是reorder再拼接最新一步生成的token
        hidden = hidden[beam_id[incomplete_idx]]  # hidden进行reorder
        prev_words = token_id[incomplete_idx].unsqueeze(1)  # (s, 1) s < beam_size
        lprob_sum = lprob_sum[
            incomplete_idx
        ]  # 这也也不需要reorder， #3那里reorder了，最后lprob_sum.shape=(beam_size,)

        if step > max_length:  # decode太长后，直接break掉
            break
        step += 1
    i = complete_seqs_scores.index(max(complete_seqs_scores))  # 寻找score最大的序列
    # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
    seq = complete_seqs[i]

    return seq


if __name__ == "__main__":
    # test for beam_search
    class RNN_(nn.Module):  # 一个假的模型
        def __init__(self):
            super().__init__()

        def forward(self, prev_words, hidden, beam_size):
            # 注意虽然模型输入其实没有beam_size，但是测试的时候这里要送进beam_size，不然会报错。
            # 也就是说测试的时候，34行代码，也就是logits, hidden = decoder()那里，稍微改一下，要送进beam_size
            seq_len, vocab_size = 12, 600
            logits = torch.randn(beam_size, seq_len, vocab_size)
            hidden = torch.rand_like(hidden) + hidden
            return logits, hidden

    m = RNN_()
    for _ in range(5):

        hidden: torch.Tensor = torch.randn(3, 128)
        beam_size: int = 3
        bos_token_idx: int = 0
        eos_token_idx: int = 2
        max_length: int = 60
        vocab_size: int = 600
        result = beam_search(m, hidden)
        print(f"result = {result}")
