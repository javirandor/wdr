import logging


class Logger:
    def __init__(self, config):
        # https://github.com/huggingface/transformers/issues/1843#issuecomment-555598281
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

        logging.basicConfig(level=logging.DEBUG)
        self.log = logging.getLogger("log")

        file_handler = logging.FileHandler("{}/out.log".format(config.model_base_path))
        file_handler.setLevel(logging.DEBUG)
        self.log.addHandler(file_handler)
