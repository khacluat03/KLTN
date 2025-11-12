from argparse import ArgumentParser

from macrec.tasks.base import Task
from macrec.dataset import amazon_process_data
from macrec.utils import init_all_seeds

class PreprocessTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--data_dir', type=str, required=True, help='input file')
        parser.add_argument('--dataset', type=str, required=True, choices=['amazon'], help='output file')
        parser.add_argument('--n_neg_items', type=int, default=7, help='numbers of negative items')
        return parser

    def run(self, data_dir: str, dataset: dir, n_neg_items: int):
        init_all_seeds(2024)
        if dataset == 'amazon':
            # suppose the base name of data_dir is the category name
            amazon_process_data(data_dir, n_neg_items)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    PreprocessTask().launch()
