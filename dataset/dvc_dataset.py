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
        num_bins: int = 100,
        max_input_tokens: int = 1000,
        max_output_tokens: int = 256,
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
                'mix garlic basil and tomatoes in a food '
                'processor',
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
            # features shape: [sequence_length, num_video_features]
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

        self.num_bins: int = num_bins
        self.max_input_tokens: int = max_input_tokens
        self.max_output_tokens: int = max_output_tokens
        self.num_text_tokens: int = len(text_tokenizer) - num_bins
        self.noise_density: float = noise_density
        self.mean_noise_span_length: int = mean_noise_span_length

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int):
        video_id: str = self.video_ids[idx]
        annotations: dict[str, Any] = self.data[video_id]
        # shape: [self.max_video_features, self.video_embedding_dim]
        video_features: torch.Tensor = self._get_video_features(video_id[-11:])
        duration = annotations["duration"]

        # Get subtitles of the current video
        has_subtitles: bool = (
            (self.video_id_to_subtitles is not None and video_id[-11:] in self.video_id_to_subtitles)
            or (self.subtitles_path is not None and os.path.exists(os.path.join(self.subtitles_path, video_id + '.pkl')))
        )
        if has_subtitles:
            if (self.video_id_to_subtitles is not None and video_id[-11:] in self.video_id_to_subtitles):
                subtitles: dict[str, Any] = self.video_id_to_subtitles[video_id[-11:]]
            else:
                assert self.subtitles_path is not None
                subtitles = pickle.load(open(os.path.join(self.subtitles_path, video_id[-11:] + '.pkl'), 'rb'))

            valid_subtitle_idxs: list[bool] = [
                (start >= 0 and end <= duration and end > start)
                for start, end in zip(subtitles["start"], subtitles["end"])
            ]
            if not any(valid_subtitle_idxs):  # no subtitles
                input_tokens = (torch.ones(1) * self.tokenizer.eos_token_id).long()
            else:
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
                time_input_tokens: list[torch.LongTensor] = [
                    torch.LongTensor(
                        [
                            self.time_tokenize(start, duration, self.num_bins),
                            self.time_tokenize(end, duration, self.num_bins)
                        ]
                    )
                    for start, end in zip(subtitles['start'], subtitles['end'])
                ]
                text_input_tokens: list[torch.LongTensor] = [
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
                input_tokens = [
                    torch.cat([time_token, text_token], dim=0)
                    for time_token, text_token in zip(time_input_tokens, text_input_tokens)
                ]
                input_tokens = torch.cat(input_tokens, 0)
                input_tokens = input_tokens[:self.max_input_tokens - 1]
                input_tokens = torch.cat(
                    [
                        input_tokens,
                        torch.LongTensor([self.tokenizer.eos_token_id])
                    ],
                    dim=0
                )
        else:
            input_tokens = (torch.ones(1) * self.tokenizer.eos_token_id).long()

        # Denoising sequence
        if len(input_tokens) > 1:
            mask_indices: np.ndarray = random_spans_noise_mask(
                len(input_tokens),
                self.noise_density,
                self.mean_noise_span_length
            )
            mask_indices = mask_indices[np.newaxis,...]
            labels_mask: np.ndarray = ~mask_indices

            mask_indices = mask_indices.astype(np.int8)
            labels_mask = labels_mask.astype(np.int8)

            input_ids_sentinel: np.ndarray = create_sentinel_ids(mask_indices, self.tokenizer, self.num_bins)
            labels_sentinel: np.ndarray = create_sentinel_ids(labels_mask, self.tokenizer, self.num_bins)

            denoising_output_tokens: torch.Tensor = torch.from_numpy(
                filter_input_ids(
                    input_ids=input_tokens.unsqueeze(0).numpy(),
                    sentinel_ids=labels_sentinel,
                    tokenizer=self.tokenizer
                )
            ).squeeze(dim=0)

            denoising_input_tokens = torch.from_numpy(
                filter_input_ids(
                    input_ids=input_tokens.unsqueeze(0).numpy(),
                    sentinel_ids=input_ids_sentinel,
                    tokenizer=self.tokenizer
                )
            ).squeeze(dim=0)

        else:
            input_tokens = torch.LongTensor([self.tokenizer.eos_token_id])
            denoising_input_tokens = torch.LongTensor([0])
            denoising_output_tokens = input_tokens

        # Dense Video Captioning/Video Chapter Generation sequence
        captions: list[str] = [self._get_text(x) for x in annotations['sentences']]
        time_output_tokens: list[torch.LongTensor] = [
            torch.LongTensor(
                [
                    self.time_tokenize(start, duration, self.num_bins),
                    self.time_tokenize(end, duration, self.num_bins)
                ]
            )
            for start, end in annotations['timestamps']
        ]
        text_output_tokens = [
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

        output_tokens = [
            torch.cat([time_token, text_token], dim=0)
            for time_token, text_token in zip(time_output_tokens, text_output_tokens)
        ]
        output_tokens = torch.cat(output_tokens, dim=0)
        output_tokens = output_tokens[:self.max_output_tokens - 1]
        output_tokens = torch.cat(
            [
                output_tokens,
                torch.LongTensor([self.tokenizer.eos_token_id])
            ],
            dim=0
        )

        return {
            "video_id": video_id,
            "duration": duration,
            "video": video_features,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "denoising_input_tokens": denoising_input_tokens,
            "denoising_output_tokens": denoising_output_tokens,
        }

    def _get_text(self, text: str) -> str:
        text = text.strip()
        text = text.capitalize()
        if text[-1] != '.':
            text = text + '.'
        return text

    def _get_video_features(self, video_id: str) -> torch.Tensor:
        if self.video_id_to_features is not None:
            assert video_id in self.video_id_to_features, video_id
            # shape: [sequence_length, num_video_features]
            video_features: torch.Tensor = self.video_id_to_features[video_id].float()
        else:
            features_path = os.path.join(self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video_features = torch.from_numpy(np.load(features_path)).float()

        num_features: int = video_features.shape[0]

        if num_features > self.max_video_features:
            sampled_features: list[torch.Tensor] = []
            for i in range(self.max_video_features):
                sampled_features.append(video_features[(i * num_features) // self.max_video_features])
            video_features = torch.stack(sampled_features)
            video_length: int = self.max_video_features
        elif num_features < self.max_video_features:
            video_length = num_features
            video_features = torch.cat(
                [video_features, torch.zeros(self.max_video_features - video_length, self.video_embedding_dim)],
                dim=0
            )
        else:
            video_length = self.max_video_features

        # shape: [self.max_video_features, self.video_embedding_dim]
        return video_features

    def time_tokenize(self, time: int, duration: int, num_bins: int) -> int:
        time_token: int = int(
           (num_bins - 1) * time / duration
        )
        assert time_token <= self.num_bins
        return time_token + self.num_text_tokens


def densevideocaptioning_collate_fn(batch):
    batch_size: int = len(batch)
    video_ids: list[str] = [batch[i]["video_id"] for i in range(batch_size)]
    duration: list[int] = [batch[i]["duration"] for i in range(batch_size)]

    video_features: torch.Tensor = torch.stack([
        batch[i]["video"] for i in range(batch_size)
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

    output_tokens = [batch[i]["output_tokens"] for i in range(batch_size)]
    max_output_length: int = max(len(tokens) for tokens in output_tokens)
    for i in range(batch_size):
        if len(output_tokens[i]) < max_output_length:
            output_tokens[i] = torch.cat(
                [
                    output_tokens[i],
                    torch.zeros(max_output_length - len(output_tokens[i])).long()
                ],
                dim=0
            )
    output_tokens = torch.stack(output_tokens)

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

    denoising_output_tokens = [batch[i]["denoising_output_tokens"] for i in range(batch_size)]
    max_denoising_output_length: int = max(len(x) for x in denoising_output_tokens)
    for i in range(batch_size):
        if len(denoising_output_tokens[i]) < max_denoising_output_length:
            denoising_output_tokens[i] = torch.cat(
                [
                    denoising_output_tokens[i],
                    torch.zeros(max_denoising_output_length - len(denoising_output_tokens[i])).long()
                ],
                dim=0
            )
    denoising_output_tokens = torch.stack(denoising_output_tokens)

    out = {
        "video_id": video_ids,
        "duration": duration,
        "video": video_features,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "denoising_input_tokens": denoising_input_tokens,
        "denoising_output_tokens": denoising_output_tokens,
    }
    return out


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
                                        num_bins=args.num_bins,
                                        max_input_tokens=args.max_input_tokens,
                                        max_output_tokens=args.max_output_tokens)
