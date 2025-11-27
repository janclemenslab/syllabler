import os
import warnings
from glob import glob
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from librosa.feature import melspectrogram

# Heavy dependencies are imported when used. Keep module import light.

warnings.filterwarnings("ignore")


def log_resize_spec(spec: np.ndarray, scaling_factor: int = 10) -> np.ndarray:
    """Log resize time axis. SCALING_FACTOR determines nonlinearity of scaling."""
    from PIL import Image  # heavy, import on demand

    resize_shape = [int(np.log(spec.shape[1]) * scaling_factor), spec.shape[0]]
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.LANCZOS))
    return resize_spec


def preprocess(
    syll,
    log_rescale=None,
    blur=None,
    flatten: bool = False,
    threshold: Optional[float] = None,
):
    import scipy.ndimage as si  # import locally

    S = syll.copy()

    if log_rescale is not None:
        for cnt, s in enumerate(S):
            S[cnt] = log_resize_spec(s)

    S_pad = np.zeros((len(S), S[0].shape[0], max([s.shape[1] for s in S])))
    for cnt, s in enumerate(S):
        if blur is not None:
            s = si.gaussian_filter(s, sigma=(1, 2))
        S_pad[cnt, :, : s.shape[1]] = s

    if flatten:
        S_pad = S_pad.reshape((S_pad.shape[0], -1))

    S_pad = np.array(S_pad)

    if threshold is not None:
        S_pad[S_pad < threshold] = threshold

    return S_pad


def make_proposals(wav_folder: str, annotations: str) -> pd.DataFrame:
    """Generate label proposals for syllables.

    Args:
        wav_folder: Folder containing .wav files.
        annotations: CSV path with columns (filename, start_seconds, stop_seconds, name).

    Returns:
        A pandas DataFrame with the same rows as the input CSV and proposed names in the 'name' column.
    """
    from sklearn.manifold import TSNE  # heavy
    from hdbscan import HDBSCAN  # heavy

    files = sorted(glob(os.path.join(wav_folder, "*.wav")))
    labels = pd.read_csv(annotations)

    hop = 64

    syll = []
    syll_name = []
    labels_index = []

    for file in files:
        audio, fs = librosa.load(file, sr=32000)
        S = melspectrogram(
            y=audio,
            sr=fs,
            S=None,
            n_fft=512,
            hop_length=hop,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="constant",
            power=2.0,
        )
        # Normalize and log
        S /= np.median(S, axis=1, keepdims=True)
        S = np.log2(S)[30:,]
        S = S[np.all(~np.isnan(S), axis=1), :]

        annot = labels[labels["filename"] == os.path.basename(file)]

        for onset, offset, name, index in zip(
            annot["start_seconds"], annot["stop_seconds"], annot["name"], annot.index
        ):
            if offset - onset == 0:
                continue
            syll.append(
                S[:, max(0, int(onset * fs / hop - 10)) : int(offset * fs / hop + 10)]
            )
            syll_name.append(name)
            labels_index.append(index)

    syll_name = np.array(syll_name)
    X = preprocess(syll, flatten=True, threshold=-1)

    run = 4
    m = TSNE(
        n_components=2, random_state=run, metric="cosine", init="pca", perplexity=30
    )
    u = m.fit_transform(X)
    # Set min_samples based on number of syllables
    nb_sylls = int(X.shape[0])
    min_samples = min(nb_sylls // 30, 8)
    if min_samples < 1:
        min_samples = 1
    hdbscan_labels = HDBSCAN(min_cluster_size=4, min_samples=min_samples).fit_predict(u)

    for idx, proposal in zip(labels_index, hdbscan_labels):
        if proposal != -1:
            labels.loc[idx, "name"] = f"L{proposal}"
    return labels
