import numpy as np

"""

        modified from:https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/hmm/hmm.py
        
        A Viterbi_decode algorithm implement by numpy
        Parameters
        ----------
        A : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or None
            The transition matrix between latent states in the HMM. Index `i`,
            `j` gives the probability of transitioning from latent state `i` to
            latent state `j`. Default is None.
        B : :py:class:`ndarray <numpy.ndarray>` of shape `(N, V)` or None
            The emission matrix. Entry `i`, `j` gives the probability of latent
            state i emitting an observation of type `j`. Default is None.
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or None
            The prior probability of each latent state. If None, use a uniform
            prior over states. Default is None.
        eps : float or None
            Epsilon value to avoid :math:`\log(0)` errors. If None, defaults to
            the machine epsilon. Default is None.
        Attributes
        ----------
        A : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)`
            The transition matrix between latent states in the HMM. Index `i`,
            `j` gives the probability of transitioning from latent state `i` to
            latent state `j`.
        B : :py:class:`ndarray <numpy.ndarray>` of shape `(N, V)`
            The emission matrix. Entry `i`, `j` gives the probability of latent
            state `i` emitting an observation of type `j`.
        N : int
            The number of unique latent states
        V : int
            The number of unique observation types
        O : :py:class:`ndarray <numpy.ndarray>` of shape `(I, T)`
            The collection of observed training sequences.
        I : int
            The number of sequences in `O`.
        T : int
            The number of observations in each sequence in `O`.
        """



def viterbi_decode(transition_matrix, emission_matrix, pi, observed_sequence):
    # transition_matrix, emission_matrix, pi是隐马尔科夫模型的三要素
    num_status = len(transition_matrix) # 状态的种类个数
    seq_len = len(observed_sequence) # T

    all_path_prob = np.zeros((num_status, seq_len))  # (N,T)  用来记录所有的路径概率的log形式
    back_pointer = np.zeros((num_status, seq_len)).astype(np.int) # 用来记录某个概率最大路径的上一个最优节点
    # t=0
    curr_observe = observed_sequence[0]
    for s in range(num_status):
        all_path_prob[s, 0] = np.log(pi[s]) + np.log(
            emission_matrix[s, curr_observe]
        )
        back_pointer[s, 0] = 0
    # t>=1
    for t in range(1, seq_len):
        curr_observe = observed_sequence[t]
        for s in range(num_status):
            curr_prob_seq = [
                all_path_prob[s_, t - 1]
                + np.log(transition_matrix[s_, s])
                + np.log(emission_matrix[s, curr_observe])
                for s_ in range(num_status)
            ]
            all_path_prob[s, t] = np.max(curr_prob_seq)
            back_pointer[s, t] = np.argmax(curr_prob_seq)

    best_path_log_prob = np.max(all_path_prob[:, seq_len - 1])
    best_pointer = np.argmax(all_path_prob[:, seq_len - 1])
    best_path = [best_pointer]

    # reverse
    for t in reversed(range(1, seq_len)):
        best_pointer = back_pointer[best_pointer, t]
        best_path.append(back_pointer[best_path[-1], t])
    best_path = best_path[::-1]

    return best_path, best_path_log_prob

#test:
# """
if __name__=="__main__":
    A=np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    B=np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    pi=np.array([0.4,0.1,0.4])
    N=3
    o=np.array([1,1,0])
    print(viterbi_decode(A,B,pi,o))
# """
