import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class LiverSequenceDataset(Dataset):
    def __init__(self, data_path, seq_len=16):
        self.data_path = data_path
        self.seq_len = seq_len
        self.samples = []

        volume_dirs = ["volume_pt1","volume_pt2","volume_pt3","volume_pt4","volume_pt5"]

        for vdir in volume_dirs:
            vpath = os.path.join(data_path, vdir)
            for file in os.listdir(vpath):
                if file.endswith(".nii"):
                    number = file.replace("volume-","").replace(".nii","")
                    volume_file = os.path.join(vpath, file)
                    seg_file = os.path.join(data_path, "segmentations", f"segmentation-{number}.nii")

                    if os.path.exists(seg_file):
                        self.samples.append((volume_file, seg_file))

        self.sequence_list = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        for volume_path, seg_path in self.samples:

            volume = nib.load(volume_path).get_fdata()
            mask = nib.load(seg_path).get_fdata()

            depth = volume.shape[2]
            tumor_slices = np.where(np.sum(mask == 2, axis=(0,1)) > 0)[0]

            # --- Tumor sequence ---
            if len(tumor_slices) > 0:
                center = tumor_slices[len(tumor_slices)//2]
                self.sequence_list.append((volume_path, seg_path, center, 1))

                # --- Healthy sequence (far from tumor) ---
                healthy_slices = [i for i in range(depth) if i not in tumor_slices]

                if len(healthy_slices) > 0:
                    healthy_center = healthy_slices[len(healthy_slices)//2]
                    self.sequence_list.append((volume_path, seg_path, healthy_center, 0))
            else:
                # Entire volume healthy
                center = depth // 2
                self.sequence_list.append((volume_path, seg_path, center, 0))

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        volume_path, seg_path, center, label = self.sequence_list[idx]

        volume = nib.load(volume_path).get_fdata()
        mask = nib.load(seg_path).get_fdata()

        volume = np.clip(volume, -200, 250)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        depth = volume.shape[2]

        start = max(0, center - self.seq_len//2)
        end = min(depth, start + self.seq_len)

        volume_seq = volume[:,:,start:end]
        mask_seq = mask[:,:,start:end]

        if volume_seq.shape[2] < self.seq_len:
            pad = self.seq_len - volume_seq.shape[2]
            volume_seq = np.pad(volume_seq, ((0,0),(0,0),(0,pad)))
            mask_seq = np.pad(mask_seq, ((0,0),(0,0),(0,pad)))

        mask_seq = (mask_seq == 2).astype(np.float32)

        volume_seq = torch.tensor(volume_seq, dtype=torch.float32)\
                        .permute(2,0,1).unsqueeze(1)

        mask_seq = torch.tensor(mask_seq, dtype=torch.float32)\
                        .permute(2,0,1).unsqueeze(1)

        label = torch.tensor(label, dtype=torch.float32)

        return volume_seq, mask_seq, label
