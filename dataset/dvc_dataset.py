import os
import torch
from torch.utils.data import Dataset
import json
import pickle
import numpy as np
from util.t5 import create_sentinel_ids, filter_input_ids, random_spans_noise_mask
from transformers import T5Tokenizer
from typing import Any


class DenseVideoCaptioning_Dataset(Dataset):
    def __init__(
        self,
        json_path: str,
        features_path: str,
        max_video_features: int = 100,
        video_embedding_dim: int = 768,
        text_tokenizer: T5Tokenizer | None = None,
        subtitles_path: str | None = None,
        total_bins: int = 100,
        max_input_tokens: int = 1000,
        max_target_tokens: int = 256,
        noise_density: float = 0.25,
        mean_noise_span_length: int = 5,
    ):

        self.data: dict[str, dict[str, Any]] = json.load(open(json_path, 'r'))
        """
        'zqTXQ-YqrgQ': {
            'duration': 181.04,
            'sentences': [
                'heat up oil garlic and herbs in a pot',
                'cut up the tomatoes',
                'mix garlic basil and tomatoes in a food processor',
                'stir in tomato paste',
                'slice the mozzeralla',
                'stretch out the dough into a circle',
                'brush oil on the dough',
                'add parmesan and tomato sauce',
                'add mozzarella cheese on top',
                'bake the pizza in the oven',
                'top with basil leaves'
            ],
            'timestamps': [
                [40, 64],
                [67, 74],
                [74, 82],
                [82, 88],
                [89, 93],
                [93, 110],
                [122, 126],
                [126, 133],
                [133, 141],
                [142, 151],
                [151, 160]
            ]
        }
        """

        self.video_ids: list[str] = list(self.data.keys())
        self.video_id_to_features: dict[str, torch.Tensor] | None = None
        self.features_path: str = None
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            # features shape: [num_video_features, video_embedding_dim]
            self.video_id_to_features = torch.load(features_path)

        self.max_video_features: int = max_video_features
        self.video_embedding_dim: int = video_embedding_dim

        self.tokenizer: T5Tokenizer | None = text_tokenizer
        assert self.tokenizer is not None

        self.video_id_to_subtitles: dict[str, dict[str, Any]] | None = None
        self.subtitles_path: str | None = None
        if subtitles_path and os.path.exists(subtitles_path) and os.path.isdir(subtitles_path):
            self.subtitles_path = subtitles_path
        elif subtitles_path and os.path.exists(subtitles_path):
            self.video_id_to_subtitles = pickle.load(open(subtitles_path, "rb"))
        else:
            print("No subtitles given or found.")
        # self.video_name_to_subtitles:
        """
        'zzT6RoI4JPU': {
            'start': [
                0.582,
                5.157,
                7.642,
                21.748
            ],
            'end': [
                5.157,
                5.699,
                12.559,
                35.0,
            ],
            'text': [
                " Enjoy today's recipe and do not forget to subscribe us.",
                "Let's cook it!",
                "Hi, here is Mom's Kitchen and welcome with Mirka van Gils.",
                "You know, little bit last days I didn't pay too much attention to my channel because I am very...'
            ]
        }
        """

        self.total_bins: int = total_bins
        self.max_input_tokens: int = max_input_tokens
        self.max_output_tokens: int = max_target_tokens
        self.num_text_tokens: int = len(text_tokenizer) - total_bins
        self.noise_density: float = noise_density
        self.mean_noise_span_length: int = mean_noise_span_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        video_id: str = self.video_ids[idx]
        annotations: dict[str, Any] = self.data[video_id]
        """
        {
            'duration': 181.04,
            'sentences': [
                'heat up oil garlic and herbs in a pot',
                'cut up the tomatoes',
                'mix garlic basil and tomatoes in a food processor',
                'stir in tomato paste',
                'slice the mozzeralla',
                'stretch out the dough into a circle',
                'brush oil on the dough',
                'add parmesan and tomato sauce',
                'add mozzarella cheese on top',
                'bake the pizza in the oven',
                'top with basil leaves'
            ],
            'timestamps': [
                [40, 64],
                [67, 74],
                [74, 82],
                [82, 88],
                [89, 93],
                [93, 110],
                [122, 126],
                [126, 133],
                [133, 141],
                [142, 151],
                [151, 160]
            ]
        }
        """

        # video_features shape: [max_video_features, video_embedding_dim]
        video_features: torch.Tensor = self._get_video_features(video_id[-11:])
        video_duration: float = annotations["duration"]

        # Check if the current video has subtitles
        has_subtitles: bool = (
            (self.video_id_to_subtitles is not None and video_id[-11:] in self.video_id_to_subtitles)
            or (self.subtitles_path is not None and os.path.exists(os.path.join(self.subtitles_path, video_id + '.pkl')))
        )
        if not has_subtitles:  # If video has no subtitles
            input_tokens = (torch.ones(1) * self.tokenizer.eos_token_id).long()  # shape: [1]
        else:  # Video has subtitles
            if (self.video_id_to_subtitles is not None and video_id[-11:] in self.video_id_to_subtitles):
                subtitles: dict[str, Any] = self.video_id_to_subtitles[video_id[-11:]]
            else:
                assert self.subtitles_path is not None
                subtitles = pickle.load(open(os.path.join(self.subtitles_path, video_id[-11:] + '.pkl'), 'rb'))
            # subtitles example
            """
            {
                'start': [
                    0.582,
                    5.157,
                    7.642,
                    21.748
                ],
                'end': [
                    5.157,
                    5.699,
                    12.559,
                    35.0,
                ],
                'text': [
                    "Enjoy today's recipe and do not forget to subscribe us.",
                    "Let's cook it!",
                    "Hi, here is Mom's Kitchen and welcome with Mirka van Gils.",
                    "You know, little bit last days I didn't pay too much attention to my channel because I am very...'
                ]
            }
            """
            valid_subtitle_idxs: list[bool] = [
                (start >= 0 and end <= video_duration and end > start)
                for start, end in zip(subtitles["start"], subtitles["end"])
            ]
            if not any(valid_subtitle_idxs):  # All subtitles are invalid
                input_tokens = (torch.ones(1) * self.tokenizer.eos_token_id).long()  # shape: [1]
            else:
                # Only keep the valid subtiles
                subtitles["start"] = [
                    start
                    for i, start in enumerate(subtitles["start"])
                    if valid_subtitle_idxs[i]
                ]
                subtitles["end"] = [
                    end
                    for i, end in enumerate(subtitles["end"])
                    if valid_subtitle_idxs[i]
                ]
                subtitles['text'] = [
                    self._get_text(text)
                    for i, text in enumerate(subtitles['text'])
                    if valid_subtitle_idxs[i]
                ]
                # Input time tokens (start time, end time) of the subtitles
                subtitles_time_input_tokens: list[torch.LongTensor] = [
                    torch.LongTensor(
                        [
                            self.time_tokenize(start, video_duration, self.total_bins),
                            self.time_tokenize(end, video_duration, self.total_bins)
                        ]
                    )
                    for start, end in zip(subtitles['start'], subtitles['end'])
                ]
                # Input text tokens of the subtitles
                subtitles_text_input_tokens: list[torch.LongTensor] = [
                    self.tokenizer(  # type: ignore
                        text,
                        add_special_tokens=False,
                        max_length=self.max_input_tokens,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors="pt"
                    )['input_ids'][0]
                    for text in subtitles['text']
                ]
                # Horizontally concatenate all the time tokens and text tokens in the subtitles
                input_tokens = []
                for time_tokens, text_tokens in zip(subtitles_time_input_tokens, subtitles_text_input_tokens):
                    input_tokens.append(
                        torch.cat([time_tokens, text_tokens], dim=0)
                    )
                input_tokens = torch.cat(input_tokens, dim=0)
                # Trim to max input tokens
                input_tokens = input_tokens[:self.max_input_tokens - 1]
                # Concatenate with the EOS token
                input_tokens = torch.cat(
                    [
                        input_tokens,
                        torch.LongTensor([self.tokenizer.eos_token_id])
                    ],
                    dim=0
                )   # shape: [num_input_tokens] (num_input_tokens <= self.max_input_tokens)

        if len(input_tokens) <= 1:  # No denoising case
            input_tokens = torch.LongTensor([self.tokenizer.eos_token_id])
            denoising_input_tokens = torch.LongTensor([0])  # shape: [1]
            denoising_target_tokens = input_tokens  # shape: [1]
        else:
            # Denoising case, mask out some spans of input sequence and try reconstructing them in the target sequence.
            # Check the T5 paper for more detail: https://arxiv.org/abs/1910.10683
            # Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
            mask_indices: np.ndarray = random_spans_noise_mask(
                len(input_tokens),
                self.noise_density,
                self.mean_noise_span_length
            )  # shape: [num_input_tokens]
            mask_indices = mask_indices[np.newaxis,...]  # shape: [1, num_input_tokens]
            labels_mask: np.ndarray = ~mask_indices

            mask_indices = mask_indices.astype(np.int8)
            labels_mask = labels_mask.astype(np.int8)

            input_ids_sentinel: np.ndarray = create_sentinel_ids(mask_indices, self.tokenizer, self.total_bins)  # shape: [1, self.max_input_tokens]
            target_ids_sentinel: np.ndarray = create_sentinel_ids(labels_mask, self.tokenizer, self.total_bins)  # shape: [1, self.max_input_tokens]

            # Mask out some spans (consecutive tokens) of the input sequence
            denoising_input_tokens = torch.from_numpy(
                filter_input_ids(
                    input_ids=input_tokens.unsqueeze(0).numpy(),
                    sentinel_ids=input_ids_sentinel,
                    tokenizer=self.tokenizer
                )
            ).squeeze(dim=0)  # shape: [num_denoising_input_tokens]

            # Reconstruct masked tokens in the target sequence
            denoising_target_tokens: torch.Tensor = torch.from_numpy(
                filter_input_ids(
                    input_ids=input_tokens.unsqueeze(0).numpy(),
                    sentinel_ids=target_ids_sentinel,
                    tokenizer=self.tokenizer
                )
            ).squeeze(dim=0)  # shape: [num_denoising_target_tokens]

        # Dense Video Captioning/Video Chapter Generation sequence
        captions: list[str] = [self._get_text(x) for x in annotations['sentences']]
        captions_time_target_tokens: list[torch.LongTensor] = [
            torch.LongTensor(
                [
                    self.time_tokenize(start, video_duration, self.total_bins),
                    self.time_tokenize(end, video_duration, self.total_bins)
                ]
            )
            for start, end in annotations['timestamps']
        ]
        captions_text_target_tokens = [
            self.tokenizer(  # type: ignore
                caption,
                add_special_tokens=False,
                max_length=self.max_output_tokens,
                padding="do_not_pad",
                truncation=True,
                return_tensors="pt"
            )['input_ids'][0]
            for caption in captions
        ]
        # Horizontally concatenate all the target time tokens and text tokens in the captions
        target_tokens = []
        for time_tokens, text_tokens in zip(captions_time_target_tokens, captions_text_target_tokens):
            target_tokens.append(
                torch.cat([time_tokens, text_tokens], dim=0)
            )
        target_tokens = torch.cat(target_tokens, dim=0)
        target_tokens = target_tokens[:self.max_output_tokens - 1]
        target_tokens = torch.cat(
            [
                target_tokens,
                torch.LongTensor([self.tokenizer.eos_token_id])
            ],
            dim=0
        )  # shape: [num_target_tokens]

        return {
            "video_id": video_id,
            "duration": video_duration,
            "video_features": video_features,                   # shape: [num_video_features, video_embedding_dim]
            "input_tokens": input_tokens,                       # shape: [num_input_tokens] (num_input_tokens <= max_input_tokens)
            "target_tokens": target_tokens,                     # shape: [num_target_tokens] (num_target_tokens <= max_target_tokens)
            "denoising_input_tokens": denoising_input_tokens,   # shape: [num_denoising_input_tokens] (num_denoising_input_tokens < num_input_tokens)
            "denoising_target_tokens": denoising_target_tokens, # shape: [num_denoising_target_tokens] (num_denoising_target_tokens < num_target_tokens)
        }

    def _get_text(self, text: str) -> str:
        text = text.strip()
        text = text.capitalize()
        if text[-1] != '.':  # Add a period after each caption
            text = text + '.'
        return text

    def _get_video_features(self, video_id: str) -> torch.Tensor:
        if self.video_id_to_features is not None:
            assert video_id in self.video_id_to_features, video_id
            # shape: [num_video_features, video_embedding_dim]
            video_features: torch.Tensor = self.video_id_to_features[video_id].float()
        else:
            features_path = os.path.join(self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video_features = torch.from_numpy(np.load(features_path)).float()

        num_video_features: int = video_features.shape[0]

        # Trim or pad the video featuures according to max_video_features
        if num_video_features > self.max_video_features:
            sampled_features: list[torch.Tensor] = []
            for i in range(self.max_video_features):
                sampled_features.append(video_features[(i * num_video_features) // self.max_video_features])
            video_features = torch.stack(sampled_features)
            video_length: int = self.max_video_features
        elif num_video_features < self.max_video_features:
            video_length = num_video_features
            video_features = torch.cat(
                [video_features, torch.zeros(self.max_video_features - video_length, self.video_embedding_dim)],
                dim=0
            )
        else:
            video_length = self.max_video_features

        # shape: [self.max_video_features, self.video_embedding_dim]
        return video_features

    def time_tokenize(self, time: int, video_duration: float, total_bins: int) -> int:
        """
        Assign a time (in seconds) into a predefined bins, in a total of `num_bins`.
        The assigned bin is then added with the total number of text tokens to get the final time token.
        """
        bin_idx: int = int(
           time * (total_bins - 1) / video_duration
        )
        assert bin_idx <= self.total_bins
        time_token: int = bin_idx + self.num_text_tokens
        return time_token


def densevideocaptioning_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size: int = len(batch)
    video_ids: list[str] = [batch[i]["video_id"] for i in range(batch_size)]
    video_durations: list[float] = [batch[i]["duration"] for i in range(batch_size)]

    video_features: torch.Tensor = torch.stack([
        batch[i]["video_features"] for i in range(batch_size)
    ])

    input_tokens = [batch[i]["input_tokens"] for i in range(batch_size)]
    max_input_length: int = max(len(tokens) for tokens in input_tokens)
    for i in range(batch_size):
        if len(input_tokens[i]) < max_input_length:
            input_tokens[i] = torch.cat(
                [
                    input_tokens[i],
                    torch.zeros(max_input_length - len(input_tokens[i])).long()
                ],
                dim=0
            )
    input_tokens = torch.stack(input_tokens)

    target_tokens = [batch[i]["target_tokens"] for i in range(batch_size)]
    max_target_length: int = max(len(tokens) for tokens in target_tokens)
    for i in range(batch_size):
        if len(target_tokens[i]) < max_target_length:
            target_tokens[i] = torch.cat(
                [
                    target_tokens[i],
                    torch.zeros(max_target_length - len(target_tokens[i])).long()
                ],
                dim=0
            )
    target_tokens = torch.stack(target_tokens)

    denoising_input_tokens = [batch[i]["denoising_input_tokens"] for i in range(batch_size)]
    max_input_length: int = max(len(x) for x in denoising_input_tokens)
    for i in range(batch_size):
        if len(denoising_input_tokens[i]) < max_input_length:
            denoising_input_tokens[i] = torch.cat(
                [
                    denoising_input_tokens[i],
                    torch.zeros(max_input_length - len(denoising_input_tokens[i])).long()
                ],
                dim=0
            )
    denoising_input_tokens = torch.stack(denoising_input_tokens)

    denoising_target_tokens = [batch[i]["denoising_target_tokens"] for i in range(batch_size)]
    max_denoising_target_length: int = max(len(x) for x in denoising_target_tokens)
    for i in range(batch_size):
        if len(denoising_target_tokens[i]) < max_denoising_target_length:
            denoising_target_tokens[i] = torch.cat(
                [
                    denoising_target_tokens[i],
                    torch.zeros(max_denoising_target_length - len(denoising_target_tokens[i])).long()
                ],
                dim=0
            )
    denoising_target_tokens = torch.stack(denoising_target_tokens)

    batch_dict: dict[str, Any] = {
        "video_id": video_ids,                              # len(video_id) = batch_size
        "video_duration": video_durations,                  # len(video_duration) = batch_size
        "video_features": video_features,                   # shape: [batch_size, num_video_features, video_embedding_dim]
        "input_tokens": input_tokens,                       # shape: [batch_size, num_input_tokens]
        "target_tokens": target_tokens,                     # shape: [batch_size, num_target_tokens]
        "denoising_input_tokens": denoising_input_tokens,   # shape: [batch_size, num_denoising_input_tokens]
        "denoising_target_tokens": denoising_target_tokens, # shape: [batch_size, num_denoising_target_tokens]
    }
    return batch_dict


def build_densevideocaptioning_dataset(dataset_name, split, args, tokenizer):
    if dataset_name == "youcook":
        if split == "train":
            json_path = args.youcook_train_json_path
        elif split == "val":
            json_path = args.youcook_val_json_path
        else:
            raise NotImplementedError
        features_path = args.youcook_features_path
        subtitles_path = args.youcook_subtitles_path
    elif dataset_name == "vitt":
        if split == "train":
            json_path = args.vitt_train_json_path
        elif split == "val":
            json_path = args.vitt_val_json_path
        elif split == "test":
            json_path = args.vitt_test_json_path
        else:
            raise NotImplementedError
        features_path = args.vitt_features_path
        subtitles_path = args.vitt_subtitles_path
    elif dataset_name == "chapters":
        if split == "train":
            json_path = args.chapters_train_json_path
        elif split == "val":
            json_path = args.chapters_val_json_path
        elif split == "test":
            json_path = args.chapters_test_json_path
        else:
            raise NotImplementedError
        features_path = args.chapters_features_path
        subtitles_path = args.chapters_subtitles_path
    else:
        raise NotImplementedError
    return DenseVideoCaptioning_Dataset(json_path=json_path,
                                        features_path=features_path,
                                        max_video_features=args.max_feats,
                                        video_embedding_dim=args.features_dim,
                                        text_tokenizer=tokenizer,
                                        subtitles_path=subtitles_path,
                                        total_bins=args.num_bins,
                                        max_input_tokens=args.max_input_tokens,
                                        max_target_tokens=args.max_output_tokens)
