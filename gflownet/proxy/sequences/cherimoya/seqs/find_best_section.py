# we are trying to find the best 5 nucleotides to change
# in the original sequence to create 
import torch
from tangermeme.utils import one_hot_encode, characters
import matplotlib.pyplot as plt
from tqdm import tqdm

def shuffle_sequence(seq):
    """
    Shuffle the sequence randomly. (this is a batch sequence)
    """
    seq = seq.clone()
    idx = torch.randperm(seq.shape[-1])
    return seq[:, :, idx]

def randomize_sequence(seq):
    """
    Randomize the sequence randomly. (this is a batch sequence)
    """
    seq = seq.clone()
    # randomize the sequence by sampling from a uniform distribution
    random_seq = torch.randint(0, 4, (seq.shape[0], seq.shape[-1]), device=seq.device)
    # convert to one-hot encoding
    random_seq_onehot = torch.zeros(seq.shape[0], 4, seq.shape[-1], device=seq.device)
    random_seq_onehot.scatter_(1, random_seq.unsqueeze(1), 1)
    return random_seq_onehot

def shuffle_original(original_seq, batch_size=10, range_of_nucleotides=5, shuffle=True):
    """
    Shuffle the original sequence randomly, but only change a certain number of nucleotides.
    """
    original_seq = original_seq.clone()
    seq_len = original_seq.shape[-1]
    
    # each batch is made up of batch * range-of-nucleotides size
    for batch_start in range(seq_len // (range_of_nucleotides * batch_size)):
        shuffled_seqs = []
        # first, build a batch of shuffled sequences
        # where each sequence starts at a different index
        batch_start = batch_start * range_of_nucleotides * batch_size
        for batch_idx in range(batch_size):
            shuffled_seq = original_seq.clone()
            idx = batch_start + range_of_nucleotides * batch_idx
            end_idx = min(idx + range_of_nucleotides, seq_len)
            if shuffle:
                shuffled_seq[:, :, idx: end_idx] = shuffle_sequence(original_seq[:, :, idx: end_idx])
            else:
                shuffled_seq[:, :, idx: end_idx] = randomize_sequence(original_seq[:, :, idx: end_idx])
            shuffled_seqs.append(shuffled_seq)
        yield torch.cat(shuffled_seqs, dim=0)

# now let's create something to plot the shape of a signal
def plot_profile(profile, save_path="profile_plot.png"):
    profile = profile.detach().cpu().numpy()
    mean_profile = np.mean(profile, axis=0).squeeze()
    x = np.arange(mean_profile.shape[0])
    plt.figure(figsize=(10, 4))
    plt.fill_between(x, mean_profile)
    plt.savefig(save_path)


if __name__ == "__main__":
    from cherimoya import Cherimoya
    import numpy as np
    from huggingface_hub import hf_hub_download

    # original_seq = one_hot_encode("ATCG")
    # shuffled_seq = shuffle_sequence(original_seq)
    # print("Shuffled sequence:", characters(shuffled_seq))
    torch.manual_seed(42)  # for reproducibility
    range_of_nucleotides = 5

    path = hf_hub_download("programmable-genomics/CATv1", "models/ENCSR000ELW/cherimoya.fold_0.torch")
    model = Cherimoya.load(path, device="cuda", compile=False).eval()

    # load the original sequence and the predicted profile
    original_seq = torch.from_numpy(np.load("./gflownet/proxy/sequences/cherimoya/seqs/X_0.npy")).to("cuda")
    pred_y_profile = torch.from_numpy(np.load("./gflownet/proxy/sequences/cherimoya/seqs/pred_y_profile.npy")).to("cuda")

    print(original_seq.shape)
    print(pred_y_profile.shape)
    batch_size = 10

    # now let's find the best section of the original sequence to shuffle
    # by finding the section with the highest MSE
    highest_mse = 0
    best_start_idx = 0
    shuffle = True
    total_idx = 0
    best_profile = None

    for batch_idx, batch in tqdm(enumerate(shuffle_original(original_seq, batch_size=batch_size, range_of_nucleotides=range_of_nucleotides, shuffle=shuffle)), total=original_seq.shape[-1] // (range_of_nucleotides * batch_size)):
        print(batch.shape)

        # let's try to shuffle the sequence and see if we can get a better score
        with torch.no_grad():
            profile_logits, log_counts = model(batch)
            profile = torch.softmax(profile_logits.float(), dim=-1)
            mse = (torch.mean((profile - pred_y_profile) ** 2, dim=(1, 2)) + 1e-8)
            if mse.max() > highest_mse:
                highest_mse = mse.max()
                highest_mse_idx = mse.argmax()
                best_start_idx = highest_mse_idx * range_of_nucleotides + range_of_nucleotides * batch_size * batch_idx
                best_profile = profile[highest_mse_idx]
            # log out all the mses to a file
            with open("mse_log.txt", "a") as f:
                for m in mse.tolist():
                    f.write(f"{total_idx * range_of_nucleotides}, {total_idx * range_of_nucleotides + range_of_nucleotides}: {m}\n")
                    total_idx += 1
    print(f"Best start index: {best_start_idx}, Highest MSE: {highest_mse.item()}")

    plot_profile(best_profile, save_path="best_shuffle_profile.png")
