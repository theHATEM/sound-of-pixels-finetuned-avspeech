import torch
import torchaudio
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import librosa, numpy as np


def _stft(self, audio):
    spec = librosa.stft(audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
    amp = np.abs(spec)
    phase = np.angle(spec)
    return torch.from_numpy(amp), torch.from_numpy(phase)


def compute_stft(audio_tensor, args):
    # audio_np = audio_tensor.cpu().numpy()

    # stft = librosa.stft(
    #     audio_np,
    #     n_fft=args.stft_frame,
    #     hop_length=args.stft_hop,
    # )
    # amp = np.abs(stft)
    # phase = np.angle(stft)

    # # Preemptively force float32 (.float()) to prevent DoubleTensor errors!
    # mag_tensor = torch.from_numpy(amp).unsqueeze(0).float()
    # phase_tensor = torch.from_numpy(phase).unsqueeze(0).float()

    # return mag_tensor, phase_tensor

    stft = torch.stft(
        audio_tensor,
        n_fft=args.stft_frame,
        hop_length=args.stft_hop,
        return_complex=True,
        pad_mode="constant",
    )
    mag = torch.abs(stft).unsqueeze(0)
    phase = torch.angle(stft).unsqueeze(0)
    return mag, phase


def music_mix_collate_fn(batch):
    batched_data = {
        "mag_mix": torch.stack([item["mag_mix"] for item in batch]),
        "frames": [
            torch.stack([item["frames"][n] for item in batch])
            for n in range(len(batch[0]["frames"]))
        ],
        "mags": [
            torch.stack([item["mags"][n] for item in batch])
            for n in range(len(batch[0]["mags"]))
        ],
    }
    if "audios" in batch[0]:
        batched_data["audios"] = [
            torch.stack([item["audios"][n] for item in batch])
            for n in range(len(batch[0]["audios"]))
        ]
        batched_data["phase_mix"] = torch.stack([item["phase_mix"] for item in batch])
        batched_data["infos"] = [item["infos"] for item in batch]

    return batched_data


class StreamingMUSICMixDataset(IterableDataset):
    def __init__(self, hf_stream, args, split="train"):
        self.hf_stream = hf_stream
        self.args = args
        self.split = split

        # Audio duration required = 65535 / 11025 ≈ 5.944 seconds
        self.audio_duration = self.args.audLen / self.args.audRate

        # We need to resize the extracted frames to 224x224 for ResNet
        # self.resize = T.Resize((self.args.imgSize, self.args.imgSize))
        self.img_transform = T.Compose(
            [
                T.Resize((self.args.imgSize, self.args.imgSize)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __iter__(self):
        mix_pool = []

        for row in self.hf_stream:
            video_decoder = row.get("video")
            audio_feature = row.get("audio")

            if video_decoder is None or audio_feature is None:
                continue

            try:
                # -----------------------------------
                # A. Metadata & Center Alignment
                # -----------------------------------
                orig_fps = video_decoder.metadata.average_fps
                duration = video_decoder.metadata.duration_seconds

                # Skip if video is shorter than the required ~5.944s of audio
                if (
                    orig_fps is None
                    or duration is None
                    or duration <= self.audio_duration
                ):
                    continue

                # Find a safe center time where the audio window won't go out of bounds
                min_center = self.audio_duration / 2.0
                max_center = duration - (self.audio_duration / 2.0)

                if self.split == "train":
                    center_time = random.uniform(min_center, max_center)
                else:
                    center_time = duration / 2.0  # Exact middle for validation

                # -----------------------------------
                # B. Extract exactly `num_frames` (1)
                # -----------------------------------
                center_frame_idx = int(center_time * orig_fps)
                indices = []

                # Handles multi-frame extraction if you ever change args.num_frames > 1
                for i in range(self.args.num_frames):
                    idx_offset = (
                        i - self.args.num_frames // 2
                    ) * self.args.stride_frames
                    # Convert target fps offset to original video fps offset
                    frame_offset = int(idx_offset * (orig_fps / self.args.frameRate))
                    idx = center_frame_idx + frame_offset
                    # Clamp index to prevent Out-Of-Bounds
                    idx = max(0, min(idx, video_decoder.metadata.num_frames - 1))
                    indices.append(idx)

                # Extract and resize frames ->[num_frames, 3, 224, 224]
                frames = video_decoder.get_frames_at(indices=indices).data

                # 🚨 NEW: Convert to float32 and scale to[0.0, 1.0]
                frames = frames.float() / 255.0

                # Apply Resize and Normalization
                frames = self.img_transform(frames)

                # -----------------------------------
                # C. Extract & Resample Audio Segment
                # -----------------------------------
                if hasattr(audio_feature, "get_all_samples"):
                    audio_tensor = audio_feature.get_all_samples().data
                    orig_sr = getattr(audio_feature, "sample_rate", 48000)
                else:
                    audio_tensor = torch.tensor(audio_feature["array"])
                    orig_sr = audio_feature["sampling_rate"]

                audio_tensor = audio_tensor.squeeze()
                if audio_tensor.ndim == 2:  # Convert to Mono
                    audio_tensor = audio_tensor.mean(
                        dim=0 if audio_tensor.shape[0] == 2 else 1
                    )

                # Slice the exact audio window centered around our frame
                start_time = center_time - (self.audio_duration / 2.0)
                start_sample = max(0, int(start_time * orig_sr))
                end_sample = start_sample + int(self.audio_duration * orig_sr)
                audio_segment = audio_tensor[start_sample:end_sample]

                # Resample to exactly 11025 Hz
                if orig_sr != self.args.audRate:
                    audio_segment = torchaudio.functional.resample(
                        audio_segment.float(), orig_sr, self.args.audRate
                    )

                # Pad or Trim to strictly match args.audLen (65535)
                if audio_segment.shape[0] > self.args.audLen:
                    audio_segment = audio_segment[: self.args.audLen]
                elif audio_segment.shape[0] < self.args.audLen:
                    pad_amount = self.args.audLen - audio_segment.shape[0]
                    audio_segment = torch.nn.functional.pad(
                        audio_segment, (0, pad_amount)
                    )

                # Compute individual STFT
                mag, phase = compute_stft(audio_segment, self.args)

                # Add to pool
                mix_pool.append(
                    {
                        "frames": frames,
                        "audio": audio_segment,
                        "mag": mag,
                        "phase": phase,
                        "info": row.get("clip_id", "unknown"),
                    }
                )

                # -----------------------------------
                # D. Mix when pool reaches `num_mix` (2)
                # -----------------------------------
                if len(mix_pool) == self.args.num_mix:
                    out_frames = [p["frames"] for p in mix_pool]
                    out_audios = [p["audio"] for p in mix_pool]
                    out_mags = [p["mag"] for p in mix_pool]
                    out_infos = [p["info"] for p in mix_pool]

                    # Mix Audios and Compute Mixed STFT
                    mixed_audio = sum(out_audios)
                    mag_mix, phase_mix = compute_stft(mixed_audio, self.args)

                    ret_dict = {
                        "mag_mix": mag_mix,
                        "frames": out_frames,
                        "mags": out_mags,
                        "audios": out_audios,  # <--- Now it's always here!
                        "phase_mix": phase_mix,
                        "infos": out_infos,
                    }

                    # yield ret_dict
                    yield ret_dict
                    mix_pool = []  # Reset for next pair

            except Exception as e:
                # ALWAYS print the error in a streaming dataset so you know if it's breaking!
                print(f"Skipping video due to error: {e}")
                continue

    def create_train_val_loader(args):
        train_stream = load_dataset(
            "ProgramComputer/avspeech-visual-audio", streaming=True, split="train"
        )
        train_stream = train_stream.shuffle(
            seed=args.seed, buffer_size=100
        ).with_format("torch")

        dataset_train = StreamingMUSICMixDataset(train_stream, args, split="train")

        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.workers,
            prefetch_factor=2,
            collate_fn=music_mix_collate_fn,
        )

        val_stream = load_dataset(
            "ProgramComputer/avspeech-visual-audio",
            streaming=True,
            split="test",
        )
        val_stream = val_stream.with_format("torch")

        dataset_val = StreamingMUSICMixDataset(val_stream, args, split="val")

        loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.workers,
            prefetch_factor=2,
            collate_fn=music_mix_collate_fn,
        )

        return loader_train, loader_val


if __name__ == "__main__":
    from arguments import ArgParser

    parser = ArgParser()
    args = parser.parse_train_arguments()

    train_stream = load_dataset(
        "ProgramComputer/avspeech-visual-audio",
        streaming=True,
        split="train[:95%]",
    )
    train_stream = train_stream.shuffle(seed=args.seed, buffer_size=100).with_format(
        "torch"
    )

    dataset_train = StreamingMUSICMixDataset(train_stream, args, split="train")

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=int(args.workers),
        prefetch_factor=2,
        collate_fn=music_mix_collate_fn,
    )

    val_stream = load_dataset(
        "ProgramComputer/avspeech-visual-audio",
        streaming=True,
        split="train[95%:]",
    )
    val_stream = val_stream.with_format("torch")

    dataset_val = StreamingMUSICMixDataset(val_stream, args, split="val")

    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=int(args.workers),
        prefetch_factor=2,
        collate_fn=music_mix_collate_fn,
    )
