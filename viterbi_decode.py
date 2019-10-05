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



def viterbi_decode(A,B,pi,N,O,eps):
    
    eps = eps

    if O.ndim == 1:
        O = O.reshape(1, -1)

    # number of observations in each sequence
    T = O.shape[1]

    # number of training sequences
    I = O.shape[0]
    if I != 1:
        raise ValueError("Can only decode a single sequence (O.shape[0] must be 1)")

    # initialize the viterbi and back_pointer matrices
    viterbi = np.zeros((N, T))
    back_pointer = np.zeros((N, T)).astype(int)

    ot = O[0, 0]
    for s in range(N):
        back_pointer[s, 0] = 0
        viterbi[s, 0] = np.log(pi[s] + eps) + np.log(B[s, ot] + eps)

    for t in range(1, T):
        ot = O[0, t]
        for s in range(N):
            seq_probs = [
                viterbi[s_, t - 1]
                + np.log(A[s_, s] + eps)
                + np.log(B[s, ot] + eps)
                for s_ in range(N)
            ]

            viterbi[s, t] = np.max(seq_probs)
            back_pointer[s, t] = np.argmax(seq_probs)

    best_path_log_prob = viterbi[:, T - 1].max()

    # backtrack through the trellis to get the most likely sequence of
    # latent states
    pointer = viterbi[:, T - 1].argmax()
    best_path = [pointer]
    for t in reversed(range(1, T)):
        pointer = back_pointer[pointer, t]
        best_path.append(pointer)
    best_path = best_path[::-1]
    return best_path, best_path_log_prob

#test:
"""

A=np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
B=np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
pi=np.array([0.4,0.1,0.4])
N=3
o=np.array([1,1,0])
eps=11e-3
print(decode(A,B,pi,N,o,eps))
"""
