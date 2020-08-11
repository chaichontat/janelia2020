from __future__ import annotations

from functools import cached_property
from typing import Literal, Tuple, Union, overload
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.core.records import ndarray
from scipy import ndimage as ndi
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from .analyzer import Analyzer

from .utils.io import hdf5_save_from_obj, sha256

Path_str = Union[str, Path]


class SpikeLoader(Analyzer):
    ARRAYS = ["spks", "imgs", "S", "imgs_stim"]
    DATAFRAMES = ["pos", "istim"]
    HYPERPARAMS = ["img_scale", "img_dim", "npzhash"]
    
    """Class to process spiking data.
    Experiment expected to include periods with stimulation by image with or without repeats.
    
    Args:
        img_scale (float, optional): Factor to scale imgs_stim. Defaults to 0.25.
    
    Attributes:
        spks (np.ndarray): (n_frame × n_neu) Spike data from suite2p.
        imgs (np.ndarray): (n_img × y × x) Image data.
        S (np.ndarray): (n_stim × n_neu): z-scored {spks} (across time) containing only stim frames.
        imgs_stim (np.ndarray): (n_img × px) z-scored flattened {imgs} (across time).
        pos (pd.DataFrame): (n_neu × [x, y]) Physical location of each neuron.
        istim (pd.Series): (n_stim) (idx_frame ↦ idx_img) with index.
        img_scale (float)
        img_dim (np.ndarray): (y × x) Scaled image size.
        npzhash (bytes): SHA-256 has of original npz file.
    
    """

    def __init__(self, img_scale: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.img_scale = img_scale

    @classmethod
    def from_npz(cls, path: Path_str, img_scale: float = 0.25) -> SpikeLoader:
        """Expected file format: npz containing the following arrays:
            - xpos (n_neu × 1): x physical location of neuron.
            - ypos (n_neu × 1): y physical location of neuron.
            - istim (n_stim × 1): (idx_stim ↦ idx_img) Maps nth (frames with stim) to img.
            - frame_start (n_stim × 1): (idx_stim ↦ idx_frame) Maps nth (frames with stim) to actual time point.
            - spks (n_neu × n_frame): Spike data from `suite2p`.
            - img (y × x × n_img): Image data.
            
        Args:
            path (Path_str)
            img_scale (float, optional): Factor to scale imgs_stim. Defaults to 0.25.

        Returns:
            SpikeLoader
        """
        
        npzhash = sha256(path)
        with np.load(path, mmap_mode="r") as npz:
            pos = pd.DataFrame({"x": npz["xpos"], "y": npz["ypos"]})
            istim = pd.Series(npz["istim"], index=npz["frame_start"])
            spks = npz["spks"].T.astype(np.float32)
            imgs = np.transpose(npz["img"], (2, 0, 1)).astype(np.float32)

        S = zscore(spks[istim.index, :], axis=0).astype(np.float32)

        def _imgs_stim():
            X = ndi.zoom(imgs[istim, ...], (1, img_scale, img_scale), order=1)
            img_dim = X.shape[1:]
            X = np.reshape(X, [len(istim), -1])
            return img_dim, zscore(X, axis=0) / np.sqrt(len(istim)).astype(np.float32)

        img_dim, imgs_stim = _imgs_stim()

        del npz, _imgs_stim
        return cls(**{k: v for k, v in locals().items() if k != "cls"})

    @overload
    def get_idx_rep(self, return_onetimers: Literal[True], stim_idx: bool) -> Tuple[ndarray, ndarray]:
        ...

    @overload
    def get_idx_rep(self, return_onetimers: Literal[False], stim_idx: bool) -> np.ndarray:
        ...

    def get_idx_rep(self, return_onetimers: bool = False, stim_idx: bool = True) -> np.ndarray:
        istim: ndarray = self.istim.array.to_numpy()
        idx_unq, unq_cnt = np.unique(istim, return_counts=True)
        idx_repeating_img = idx_unq[np.argwhere(unq_cnt > 1)]
        idx = -1 * np.ones(
            [len(idx_repeating_img), np.max(unq_cnt)], dtype=istim.dtype
        )  # Generate array with (n_repeated_stim, max_repeats).
        
        for i in range(len(idx_repeating_img)):
            if stim_idx:
                curr = np.where(istim == idx_repeating_img[i])[0]  # From integer index.
            else:
                curr = (
                    self.istim.where(istim == idx_repeating_img[i]).dropna().index
                )  # From index.
            idx[i, : curr.size] = curr

        if return_onetimers:  # TODO For stim_idx=True only.
            idx_one = np.where(
                np.isin(np.arange(len(istim)), np.array(idx).flatten(), invert=True)
            )[0]
            return idx, idx_one

        return idx

    @cached_property
    def idx_spont(self) -> np.ndarray:
        if self.spks is None:
            raise ValueError("Need to load full data for this")
        idx_spont = np.where(
            np.isin(
                np.arange(np.max(self.istim.index) + 1),
                self.istim.index,
                assume_unique=True,
                invert=True,
            )
        )[0]  # Invert indices.
        
        assert idx_spont.size + self.istim.index.size == self.spks.shape[0]
        return idx_spont

    def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
        return train_test_split(
            self.imgs_stim, self.S, test_size=test_size, random_state=random_state
        )

    def save_processed(self, path: Path_str, **kwargs) -> None:
        """Save only processed data.
        Discard {spks} and {imgs}.

        Args:
            path (Path_str):

        """
        self.imgs_stim
        self.S
        arrs = ["imgs_stim", "S"]
        return hdf5_save_from_obj(
            path,
            group="SpikeLoader",
            obj=self,
            arrs=arrs,
            dfs=self.DATAFRAMES,
            params=self.HYPERPARAMS,
            **kwargs
        )


def convert_x(xpos, offset=5, width=473, gap=177):  # Total 650.
    x = xpos.copy()
    x -= offset
    z = x // (width + gap)
    x = x % (width + gap)
    return x, z


def gen_test_data(path_in: str, path_out: str) -> None:
    # SHA256 of ori file: be94f5c531a47499c0010785402bf003bfdf6c456202b53070759fde73200a36
    npz = dict(np.load(path_in))
    npz["spks"] = npz["spks"][:500]  # Get first 500 neurons.
    np.savez(path_out, **npz)
    
    loader = SpikeLoader.from_npz(path_out)
    loader.imgs = None
    loader.save(path_out[:-4] + ".hdf5", complevel=9, overwrite=True)


# if __name__ == "__main__":
    # gen_test_data("data/superstim.npz", "data/test.npz")
