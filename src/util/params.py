"""
    Parameters associated with each version used
"""

import argparse


class Params():
    init_params = {
        "folder": ["v1", "v2", "v3"],
        "add_tag": [False, True, True],
        "domain_tag": [False, False, True],
        "word_embed_name": ["train_word2vec.model", "train_word2vec.model", "train_fasttext.model"],
        "isWord2vec": [True, True, False],

        # String params only replaces the first value by folder
        "dump_folder": "dumps/{}/",
        "checkpoint_folder": "./checkpoints/{}/",
        "data_folder": "data/{}/"
    }

    def __init__(self, init_parser = None):
        if init_parser:
            parser = init_parser
        else:
            parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--version', type=int, default=1)
        self.args = parser.parse_args()

        i = self.args.version - 1

        self.folder = Params.init_params["folder"][i]
        self.add_tag = Params.init_params["add_tag"][i]
        self.domain_tag = Params.init_params["domain_tag"][i]
        self.word_embed_name = Params.init_params["word_embed_name"][i]
        self.is_word2vec = Params.init_params["isWord2vec"][i]

        self.dump_folder = Params.init_params["dump_folder"].format(self.folder)
        self.checkpoint_folder = Params.init_params["checkpoint_folder"].format(self.folder)
        self.data_folder = Params.init_params["data_folder"].format(self.folder)

