from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import os

from fuzzywuzzy import fuzz

# from titlecase import titlecase


class Generator(object):
    def __init__(
        self, model_dir="models", init_epoch=70,
    ):

        # where to get the model files?
        self.model_dir = model_dir
        self.init_epoch = init_epoch
        self.pretrained_path = self._construct_pretrained_path(
            self.model_dir, self.init_epoch
        )

        # model's start and end tokens
        self.START_TKN = "<|startoftext|>"
        self.END_TKN = "<|endoftext|>"

        # load the model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")

        # select torch device (cpu/gpu)
        self.device = self._select_device()
        self.model = self.model.to(self.device)

    @staticmethod
    def _construct_pretrained_path(model_dir, epoch):
        ptp = os.path.join(model_dir, f"distilgpt2_onion_{epoch}.pt")
        assert os.path.exists(ptp), "file DNE"
        return ptp

    @staticmethod
    def _select_device():
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        return device

    def _load_weights_from_file(self, path):
        self.model.load_state_dict(torch.load(path))

    def _load_weights_from_epoch(self, epoch: int):
        path = self._construct_pretrained_path(self.model_dir, epoch)
        self.model.load_state_dict(torch.load(path))

    @staticmethod
    def _select_a_top_token(slogits, max_candidates):
        index = np.argpartition(slogits, -max_candidates)[-max_candidates:]

        top_slogits = slogits[index]
        top_slogits = top_slogits / np.sum(top_slogits)  # Normalize

        choice = np.random.choice(max_candidates, 1, p=top_slogits)
        token_id = index[choice][0]
        return int(token_id)

    def generate(
        self,
        seed="",
        max_tokens=100,
        token_cands=40,
        token_cands_init=800,
        init_thresh=3,
    ):
        starting_text = self.START_TKN + seed.lower()
        self.model.eval()
        with torch.no_grad():
            text_vector = (
                torch.tensor(self.tokenizer.encode(starting_text))
                .unsqueeze(0)
                .to(self.device)
            )
            orig_shape = text_vector.shape
            for token_idx in range(max_tokens):
                outputs = self.model(text_vector, labels=text_vector)
                _, logits = outputs[:2]
                only_batch_last_embedding = logits[0, -1]
                softmax_logits = torch.softmax(only_batch_last_embedding, dim=0)

                # in the beginning, use more tokens to ensure novel results
                if token_idx + orig_shape[1] <= init_thresh + 1:
                    max_token_cands = token_cands_init
                else:
                    max_token_cands = token_cands

                next_token_id = self._select_a_top_token(
                    softmax_logits.to("cpu").numpy(), max_token_cands
                )
                text_vector = torch.cat(
                    [
                        text_vector,
                        torch.ones((1, 1)).long().to(self.device) * next_token_id,
                    ],
                    dim=1,
                )  # Add the last word to the running sequence

                if next_token_id in self.tokenizer.encode(self.END_TKN):
                    break

            output_list = list(text_vector.squeeze().to("cpu").numpy())
            output_text = self.tokenizer.decode(output_list)

        return output_text

    def generate_clean(
        self,
        seed="",
        max_tokens=100,
        token_cands=40,
        token_cands_init=800,
        init_thresh=3,
    ):
        text = self.generate(
            seed, max_tokens, token_cands, token_cands_init, init_thresh
        )
        text = text.replace(self.START_TKN, "").replace(self.END_TKN, "")
        # TODO: add proper title case
        return text

    def set_epoch(self, epoch):
        self._load_weights_from_epoch(epoch)


class Comparisons(object):
    def __init__(self, data_file):
        self.data_file = data_file

    @staticmethod
    def _sort_and_trim(matches, n):
        matches_ = sorted(matches, key=lambda x: -x["score"])
        return matches_[:3]

    def compare(self, text, n_closest=3):
        top_matches = []
        output = text.lower()
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                realdata = line[:-1].lower()
                sim = fuzz.token_sort_ratio(output, realdata)
                top_matches.append({"text": realdata, "score": sim})
                top_matches = self._sort_and_trim(top_matches, n=n_closest)
        return top_matches
