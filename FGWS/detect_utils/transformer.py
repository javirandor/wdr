from copy import deepcopy
from utils import inference, list_join
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch.nn as nn
import torch


class Transformer:
    def __init__(
        self, orig_text, model, detector, data_module, tokenizer, config, is_huggingface=True, bert_wrapper=None
    ):
        self.orig_text = orig_text
        self.model = model
        self.tokenizer = tokenizer
        self.detector = detector
        self.data_module = data_module
        self.config = config
        self.bert_wrapper = bert_wrapper
        self.is_huggingface = is_huggingface

        self.orig_pred = None
        self.orig_prob = None
        self.transformed_text = None
        self.transformed_pred = None
        self.transformed_prob = None
        self.transformed_reps = None
        self.diff = None

        self.apply_transformation()

    def inference(self, text):
        assert isinstance(text, list)

        softmax = nn.Softmax(dim=1)

        if self.is_huggingface:
            a = self.tokenizer(TreebankWordDetokenizer().detokenize(text), return_tensors="pt", padding=True, truncation=True)
            pred = self.model(**a).logits
        else:
            tokens = self.model.tokenizer([TreebankWordDetokenizer().detokenize(text)])
            pred = self.model(torch.tensor(tokens))

        outputs = softmax(pred)
        probs = outputs.cpu().detach().numpy().tolist()
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().detach().numpy().tolist()

        preds, probs = preds[0], probs[0]

        return preds, probs


    def apply_transformation(self):
        self.orig_pred, self.orig_prob = self.inference(self.orig_text)
        self.transformed_text, self.transformed_reps = self.detector.detector_module(
            deepcopy(self.orig_text)
        )
        self.transformed_pred, self.transformed_prob = self.inference(
            self.transformed_text
        )
        self.diff = max(
            0, self.orig_prob[self.orig_pred] - self.transformed_prob[self.orig_pred]
        )

    def flipped(self, gamma):
        return self.diff > gamma

    def print_info(self, logger, gamma, adversarial=None):
        logger.log.info(
            "{}:".format("Adversarial" if adversarial is not None else "Original")
        )
        logger.log.info("{}".format(list_join(self.orig_text)))
        logger.log.info("Pred: {} (prob: {})".format(self.orig_pred, self.orig_prob))

        if adversarial is not None:
            logger.log.info("Adversarial substitutions: {}".format(adversarial))

        logger.log.info(
            "Transformed {}:".format(
                "adversarial" if adversarial is not None else "original"
            )
        )
        logger.log.info("{}".format(list_join(self.transformed_text)))
        logger.log.info(
            "Pred: {} (prob: {})".format(self.transformed_pred, self.transformed_prob)
        )
        logger.log.info(
            "Transformed replacements: {}".format(sorted(self.transformed_reps))
        )
        logger.log.info("gamma={}".format(gamma))
        logger.log.info(
            "dist={}=max(0, {}-{})".format(
                self.diff,
                self.orig_prob[self.orig_pred],
                self.transformed_prob[self.orig_pred],
            )
        )
