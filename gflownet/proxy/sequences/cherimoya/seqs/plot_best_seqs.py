import matplotlib.pyplot as plt
import os

# now let's create something to plot the shape of a signal
def plot_profile(profile, save_path="profile_plot.png"):
    profile = profile.detach().cpu().numpy()
    mean_profile = np.mean(profile, axis=0).squeeze()
    x = np.arange(mean_profile.shape[0])
    plt.figure(figsize=(10, 4))
    plt.fill_between(x, mean_profile)
    plt.savefig(save_path)


if __name__ == '__main__':
    from cherimoya import Cherimoya
    import numpy as np
    from huggingface_hub import hf_hub_download
    import torch

    path = hf_hub_download("programmable-genomics/CATv1", "models/ENCSR000ELW/cherimoya.fold_0.torch")
    model = Cherimoya.load(path, device="cuda", compile=False).eval()

    # load the original sequence and the predicted profile
    original_seq = torch.from_numpy(np.load("./gflownet/proxy/sequences/cherimoya/seqs/X_0.npy")).to("cuda")
    pred_y_profile = torch.from_numpy(np.load("./gflownet/proxy/sequences/cherimoya/seqs/pred_y_profile.npy")).to("cuda")

    # get this from find_best_section.py
    start_idx = 1440
    range_of_nucleotides = 5

    # read in a bunch of the sequences
    seqs = [
        "CAAAT",
        "ATGTA"
    ]

    edited_seqs = []
    from tangermeme.utils import one_hot_encode, characters
    for seq in seqs:
        onehot_seq = one_hot_encode(seq).to("cuda")
        edited_seq = original_seq.clone()
        edited_seq[:, :, start_idx:start_idx + range_of_nucleotides] = onehot_seq
        edited_seqs.append(edited_seq)

    with torch.no_grad():
        edited_seqs = torch.cat(edited_seqs, dim=0)
        profile_logits, log_counts = model(edited_seqs)
        profile = torch.softmax(profile_logits.float(), dim=-1)
        mses = torch.mean((profile - pred_y_profile) ** 2, dim=(1, 2))

    os.makedirs("plots", exist_ok=True)
    for profile, seq, mse in zip(profile, seqs, mses):
        plot_profile(profile, save_path=f"plots/profile_{seq}.png")
        # write out the MSE as well
        with open(f"plots/mse_{seq}.txt", "w") as f:
            f.write(f"MSE: {mse.item()}\n")
