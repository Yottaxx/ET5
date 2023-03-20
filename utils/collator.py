import torch
from transformers import DataCollatorForSeq2Seq


class DataCollatorForSeq2SeqEntailment(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    encoder_classifier: int = -100

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        ce_flag = "ce_label" in features[0].keys()

        if ce_flag:

            num_flatten = len(features[0]["input_ids"])

            flattened_features = [
                [{k: v[i] for k, v in feature.items()} for i in range(num_flatten)] for feature in features
            ]
            features = sum(flattened_features, [])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        rule_idx = [torch.tensor(feature["rule_idx"]) for feature in features] if "rule_idx" in features[0].keys() else None
        user_idx = [torch.tensor(feature["user_idx"]) for feature in features] if "user_idx" in features[0].keys() else None
        relation = [feature["relationInput"] for feature in features] if "relationInput" in features[0].keys() else None

        if rule_idx is not None and user_idx is not None:
            for feature in features:
                try:
                    feature.pop("rule_idx")
                    feature.pop("user_idx")
                    feature.pop("relationInput")

                except KeyError:
                    pass


        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        entailment_mask = [feature["entailment_mask"] for feature in features] if "entailment_mask" in features[
            0].keys() else None
        entailment_label = [feature["entailment_label"] for feature in features] if "entailment_label" in features[
            0].keys() else None
        entailment_len = [feature["entailment_len"] for feature in features] if "entailment_len" in features[
            0].keys() else None

        edu_entailments_attention_mask = []

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        ## add edu label
        if self.encoder_classifier == 1:
            if entailment_mask is not None:

                max_e_mask_length = max(len(l) for l in entailment_mask)
                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [0] * (max_e_mask_length - len(feature["entailment_mask"]))

                    feature["entailment_mask"] = (
                        feature["entailment_mask"] + remainder if padding_side == "right" else remainder + feature[
                            "entailment_mask"]
                    )

                max_e_mask_sum = max(sum(l) for l in entailment_mask)

                for feature in features:
                    remainder = [1] * (max_e_mask_sum - sum(feature["entailment_mask"]))

                    sum_temp_before = sum(feature["entailment_mask"])

                    len_stay = len(feature["entailment_mask"]) - len(remainder)
                    feature["entailment_mask"] = (
                            feature["entailment_mask"][:len_stay] + remainder
                    )
                    sum_temp = sum(feature["entailment_mask"])

                    attention_mask_temp = [False] * sum_temp_before + [True] * len(remainder)

                    assert len(attention_mask_temp) == sum_temp
                    assert sum(attention_mask_temp) == (sum_temp - sum_temp_before)

                    edu_entailments_attention_mask.append(attention_mask_temp)

                    assert max_e_mask_sum == sum_temp

            if entailment_label is not None:
                max_e_mask_sum = max(entailment_len)
                max_e_label_length = max(len(l) for l in entailment_label)

                assert max_e_label_length == max_e_mask_sum
                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [-100] * (max_e_label_length - len(feature["entailment_label"]))
                    feature["entailment_label"] = (
                        feature["entailment_label"] + remainder if padding_side == "right" else remainder + feature[
                            "entailment_label"]
                    )
        else:
            for feature in features:
                try:
                    feature.pop("entailment_mask")
                    feature.pop("entailment_label")
                    feature.pop("encoder_label")
                    feature.pop("entailment_len")
                except KeyError:
                    pass

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        if self.encoder_classifier == 1:
            features['rule_mask'] = (features['entailment_label'] != -100)
            features['entailment_len'] = (torch.tensor(entailment_len))
        features['edu_attention_mask'] = (torch.tensor(edu_entailments_attention_mask))

        if ce_flag:
            features['ce_label'] = (torch.tensor([0] * int(len(features['input_ids'])/2)))

        features['rule_idx'] = rule_idx
        features['user_idx'] = user_idx
        features['relationInput'] = relation

        return features


class DataCollatorForCE(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        ce_loss = [0] * len(features)
        if isinstance(features[0]["train"], int):
            trainFlag = 0
        else:
            trainFlag = 1
            num_flatten = len(features[0]["train"])

        if trainFlag:
            flattened_features = [
                [{k: v[i] for k, v in feature.items()} for i in range(num_flatten)] for feature in features
            ]
            features = sum(flattened_features, [])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        if trainFlag:
            features['ce_label'] = torch.tensor(ce_loss)
        return features
