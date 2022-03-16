"""
Experiment Dataset API
"""
from difflib import SequenceMatcher
import pandas as pd
import json
import hashlib
import os
from functools import lru_cache

def get_hash(original_string):
    result = hashlib.sha256(original_string.encode())
    return result.hexdigest()

class HybridDialogueDataset():
    """
    Dataset API for conversations, turns, and candidates
    mode: 'train' 'validate' 'test'
    """

    def __init__(self):
        """ loads the data """
        all_data_file_path = 'experimental_data.json'
        with open(all_data_file_path, 'r') as f:
            self.all_data = json.load(f)

        # training, validation, and testing splits
        all_conversation_keys = list(self.all_data['conversations'].keys())
        sp1 = int(0.9 * len(all_conversation_keys))
        sp2 = int(0.95 * len(all_conversation_keys))

        self.conversation_keys = {
            'train': all_conversation_keys[:sp1],
            'validate': all_conversation_keys[sp1:sp2],
            'test': all_conversation_keys[sp2:]
        }

        self.all_conversations = self.all_data['conversations']
        self.all_candidates = self.all_data['all_candidates']
        self.turns = self.all_data['qas']

        self.ott_data_dir = '../../Conv_generated_json_files/'

    def get_all_candidates(self):
        """
        Returns all the candidates
        {cand_id: cand_obj}
        """
        return self.all_candidates

    def get_candidate(self, cand_id):
        """ 
        Gets a candidate data from cand_id 
        return {
            'the_type': the_type: paragraph || table || row || cell
            'raw_content': content,
            'linearized_input': linearized_input,
            'row': row,
            'col': col,
            'page_key': page_key,
            'table_key': table_key
        }
        """
        return self.all_candidates[cand_id]

    def get_conversations(self, mode):
        """ 
        Gets all conversations from a mode 
        Returns { conversation_id1: [turn_id1, turn_id2, turn_id3], ...}
        """
        return {key: self.all_conversations[key] for key in self.conversation_keys[mode]}

    def get_turn(self, turn_id):
        """
        Gets the turn with turn_id
        returns {
            'conversation_id': conversation_id,
            'current_query': current_query,
            'current_cands_ids': current_cands_ids,
            'possible_next_cands_ids': possible_next_cands_ids,
            'correct_next_cands_ids': correct_next_cands_ids,
            'short_response_to_query': short_response,
            'long_response_to_query': conversational_response,
            'position': utterance_idx // 2
        }
        """
        return self.turns[turn_id]

    def get_turns(self, mode):
        """
        Gets all the expanded turns from a mode
        Returns [turn_id1: {current_query, possible_next_cands_ids, ...}, turn_id2: ...]
        """
        turn_ids = self.get_turn_ids(mode)
        return {turn_id: self.get_turn(turn_id) for turn_id in turn_ids}

    def get_turn_ids(self, mode):
        """ 
        Gets all turns from a mode 
        Returns [turn_id1, turn_id2, ...]
        """
        turns = []
        for conversation_key in self.conversation_keys[mode]:
            for turn in self.all_conversations[conversation_key]:
                turns.append(turn)
        return turns

    def get_page_data(self, key):
        # Internal
        file_name = self.ott_data_dir + get_hash(key) + '.json'
        # check page doens't exist in the dataset
        if not os.path.isfile(file_name):
            # print('page DNE', key)
            return True, ''
        with open(file_name) as f:
            page_data = json.load(f)
        is_only_passage = type(page_data) != list

        if not is_only_passage:
            page_data = {
                "tables": page_data,
            }
        return is_only_passage, page_data

    def get_intro_from_page_key(self, page_key):
        """ 
        Returns the intro paragraph from a page with page_key
        """
        is_only_passage, page_data = self.get_page_data(page_key)
        if not page_data:
            print('no page data for ', page_key)
            return ''
        if is_only_passage:
            return page_data['passage']
        else:
            return page_data['tables'][0]['intro']
    
    @lru_cache(maxsize=10)
    def get_table_data(self, table_key, expand_links=True):
        """
        Returns tuple (pandas df, 2d list) of the table with table_key
        expand_links replaces all links with their linked paragraphs
        """
        arr = table_key.rsplit('_', 1)
        page_key, table_num = arr[0], arr[1]

        assert(table_num.isdigit())
        table_num = int(table_num)

        is_only_passage, page_data = self.get_page_data(page_key)
        assert not is_only_passage

        # itterate through each table and find the table that matches the table key
        for table in page_data['tables']:
            if table['uid'] == table_key:
                found_match = True
                break
        
        assert found_match

        table_data = []
        for row in table['data']:
            row_data = []
            for cell in row:
                # one cell may have multiple URLs so we loop thru each
                # group of (txt, url)
                txts = cell[0]
                urls = cell[1]
                cell_txt = ""
                for txt, url in zip(txts, urls):
                    if url and expand_links:
                        # expanding the cell txt to include linked passage text
                        linked_page_key = url[6:]  # removing prefix /wiki/
                        linked_paragraph = self.get_intro_from_page_key(linked_page_key)
                        cell_txt += txt + ": " + linked_paragraph + "; "
                    else:
                        cell_txt += txt + "; "
                cell_txt = cell_txt[:-2] # remove the " , " at the end of the string
                row_data.append(cell_txt)
            table_data.append(row_data)
        headers = [' '.join(cell[0]) for cell in table['header']]
        df = pd.DataFrame(table_data, columns=headers)
        return df, table_data

    def get_cell_data(self, row_idx, col_idx, table_key, expand_links=True):
        """
        Returns the cell value at the coordinates of the table with table_key
        """
        _, table_data = self.get_table_data(table_key, expand_links)
        return table_data[row_idx][col_idx]