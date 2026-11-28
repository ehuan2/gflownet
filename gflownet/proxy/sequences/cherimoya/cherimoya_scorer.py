"""
Reward for the Toy DNA Sequence scorer environment.

This scores things according to the weight in the nucleotide score dictionary.
Naturally, we will likely want to have all of the highest scoring nucleotides
will likely be the ones that are the result of the reward function itself.
"""

import numpy as np
import torch
import ot
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class CherimoyaScorer(Proxy):
    def __init__(
        self,
        batch_size: int = 50,
        **kwargs,
    ):
        """
        Parameters
        ----------
        nucleotide_scores : dict
            The score for each nucleotide.
        """
        super().__init__(**kwargs)
        from huggingface_hub import hf_hub_download
        from cherimoya import Cherimoya

        # Let's grab the trained model
        # ENCSR000ELW -- let's run this one!
        path = hf_hub_download("programmable-genomics/CATv1", "models/ENCSR000ELW/cherimoya.fold_0.torch")
        self.model = Cherimoya.load(path, device="cuda", compile=False).eval()
        # also load the y profile that we want to get
        self.pred_y_profile = torch.from_numpy(
            np.load("./gflownet/proxy/sequences/cherimoya/seqs/pred_y_profile.npy")
        ).to("cuda")
        self.original_seq = torch.from_numpy(
            np.load("./gflownet/proxy/sequences/cherimoya/seqs/X_0.npy")
        ).to("cuda")
        # We should get this from the gflownet/proxy/sequences/cherimoya/seqs/find_best_section.py
        # this will depend on how large we want to edit -- configurable in range_of_nucleotides 
        self.start_idx = 1440

    def onehot2idx(self, onehot_seq: TensorType["batch", 4, "seq_len"]) -> TensorType["batch", "seq_len"]:
        """
        Convert a one-hot encoded sequence to an index-based sequence.
        """
        # add 1 because of the PAD token
        return torch.argmax(onehot_seq, dim=1) + 1

    def print_sequence(self, seq):
        # use tangermeme to convert instead, don't trust the GFlowNet way.
        from tangermeme.utils import characters   # Convert a one-hot encoding back into a string
        print(f'Sequence: {characters(seq)}')

    def score_sequence(self, state_seqs: TensorType["batch", 4, "seq_len"]) -> TensorType["batch"]:
        """
        Score the sequence according to the predicted track differences.
        """
        # now for the actual scoring, we will edit the original sequence with
        # the current sequence at the start index, and then run the model
        edited_seqs = self.original_seq.clone().repeat(state_seqs.shape[0], 1, 1)
        edited_seqs[:, :, self.start_idx : self.start_idx + state_seqs.shape[-1]] = state_seqs.clone()

        profile_logits, log_counts = self.model(edited_seqs.to("cuda"))
        profile = torch.softmax(profile_logits.float(), dim=-1)
        # we can totally just do MSE between these two profiles!
        # the higher the MSE, the worse the score, so let's take the 1/MSE
        # so that way it's still positive (with an epsilon added)
        # take the log of it as well
        rewards = torch.log(1.0 / (torch.mean((profile - self.pred_y_profile) ** 2, dim=(1, 2)) + 1e-8))

        # but give a large negative reward as well for the original sequence so that it doesn't sample that
        original_sec = self.original_seq[:, :, self.start_idx : self.start_idx + state_seqs.shape[-1]].to(self.device)
        # have to check if the original sequence is equal to the state
        # so we have to go through all the batches and see if both dimensions are equal
        rewards[(original_sec == state_seqs).all(dim=(1, 2))] = 0.0
        return rewards.to(self.device)

    def __call__(self, states: TensorType["batch", 1]) -> TensorType["batch"]:
        states = torch.as_tensor(states, device=self.device, dtype=self.float)

        rewards = torch.zeros(states.shape[0], device=self.device, dtype=self.float)

        # let's reshape the states to fit what we want
        states = states.reshape(states.shape[0], -1, 5)
        # remove the PAD token and permute to (batch, 4, seq_len)
        states = states[:, :, 1:].permute(0, 2, 1)
        # need to batch this 
        rewards = torch.tensor([], device=self.device, dtype=self.float)
        for i in range(0, states.shape[0], 50):
            batch_states = states[i:i+50]
            with torch.no_grad():
                batch_rewards = self.score_sequence(batch_states)
            rewards = torch.cat((rewards, batch_rewards), dim=0)
        return rewards
