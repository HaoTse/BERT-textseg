from __future__ import absolute_import
import re
from pathlib2 import Path
from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer

import utils.wiki_utils as wiki_utils
from utils.my_logging import logger


class Examples():
    def __init__(self, high_granularity=False):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

        self.high_granularity = high_granularity
    
    def get_example(self, root, dataset='None', folder=False):
        if (folder):
            textfiles = self._get_files(root)
        else:
            root_path = Path(root)
            cache_path = self._get_cache_path(root_path)
            if not cache_path.exists():
                self._cache_wiki_filenames(root_path)
            textfiles = cache_path.read_text().splitlines()

        if len(textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))

        pt_data = []
        for textfile in tqdm(textfiles, desc=f'[{dataset}] '):
            srcs, tgts = self._read_wiki_file(Path(textfile), ignore_list=True)
            
            for src, tgt in zip(srcs, tgts):
                # transfer to bert data
                src_txt = [' '.join(sent) for sent in src]
                src_txt = ' [SEP] [CLS] '.join(src_txt)
                src_subtokens = self.tokenizer.tokenize(src_txt)
                src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

                # deal special token
                if len(src_subtokens) > 512:
                    logger.info('Contain special token' + ' '.join(src_subtokens))
                    continue
                src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

                _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
                segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
                segments_ids = []
                for i, s in enumerate(segs):
                    if (i % 2 == 0):
                        segments_ids += s * [0]
                    else:
                        segments_ids += s * [1]
                cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

                assert len(tgt) == len(cls_ids)

                pt_data.append({'src_txt': src, 'src_idx': src_subtoken_idxs, 'labels': tgt, 'segments_ids': segments_ids, 'cls_ids': cls_ids})

        return pt_data

    def _get_files(self, path):
        all_objects = Path(path).glob('**/*')
        files = [str(p) for p in all_objects if p.is_file()]
        return files

    def _get_cache_path(self, wiki_folder):
        cache_file_path = wiki_folder / 'paths_cache'
        return cache_file_path

    def _cache_wiki_filenames(self, wiki_folder):
        files = Path(wiki_folder).glob('*/*/*/*')
        cache_file_path = self._get_cache_path(wiki_folder)

        with cache_file_path.open('w', encoding='utf-8') as f:
            for file in files:
                f.write(str(file) + u'\n')

    def _clean_section(self, section):
        cleaned_section = section.strip('\n')
        return cleaned_section

    def _get_scections_from_text(self, txt):
        sections_to_keep_pattern = wiki_utils.get_seperator_foramt() if self.high_granularity else wiki_utils.get_seperator_foramt(
            (1, 2))
        if not self.high_granularity:
            # if low granularity required we should flatten segments within segemnt level 2
            pattern_to_ommit = wiki_utils.get_seperator_foramt((3, 999))
            txt = re.sub(pattern_to_ommit, "", txt)

            #delete empty lines after re.sub()
            sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
            txt = '\n'.join(sentences).strip('\n')


        all_sections = re.split(sections_to_keep_pattern, txt)
        non_empty_sections = [s for s in all_sections if len(s) > 0]

        return non_empty_sections

    def _get_sections(self, path):
        file = open(str(path), "r")
        raw_content = file.read()
        file.close()

        clean_txt = raw_content.strip()

        sections = [self._clean_section(s) for s in self._get_scections_from_text(clean_txt)]

        return sections

    def _read_wiki_file(self, path, remove_preface_segment=True, ignore_list=False):
        ret_data = []
        ret_targets = []
        data = []
        targets = []
        all_sections = self._get_sections(path)
        required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
        required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]

        for section in required_non_empty_sections:
            sentences = section.split('\n')
            
            # check length of sentences
            total_size = sum(len(x) for x in sentences) + 2 * (len(sentences))
            if total_size > 512: # max lenght of BERT is 512
                logger.info('Sentences total length is too long')
                continue

            if sentences:
                if ignore_list:
                    sentences = [sent for sent in sentences if wiki_utils.get_list_token() + "." != sent]

                # check data length
                if sum(len(x) for x in data) + 2 * len(data) + total_size > 512:
                    ret_data.append(data)
                    ret_targets.append(targets)
                    data = []
                    targets = []
                
                for sentence in sentences:
                    if len(sentence) > 0:
                        data.append(sentence)
                        targets.append(0)
                    else:
                        logger.info('Sentence in wikipedia file is empty')
                if targets:
                    targets[-1] = 1

        return ret_data, ret_targets
    