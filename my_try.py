import gensim.downloader as api
import re
from bidaf import BiDAF
import os
import csv


class MyWikiIterable:
    def __init__(self, iter_type, fpath):
        self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence

        Replace this function if you want to handle preprocessing differently"""
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def __iter__(self):
        # Defining some consants for .tsv reading
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        LABEL_INDEX = 6

        document_group = []
        label_group = []

        n_relevant_docs = 0
        n_filtered_docs = 0

        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                    else:
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []

            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0

class MyOtherWikiIterable:
    """Slight modification over old one for testing"""
    def __init__(self, fpath):
        # self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        # self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)
        self.to_print = ""

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence

        Replace this function if you want to handle preprocessing differently"""
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def  get_stuff(self):
        # Defining some consants for .tsv reading
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        ANSWER_ID_INDEX = 4
        LABEL_INDEX = 6

        document_group = []
        label_group = []

        n_relevant_docs = 0
        n_filtered_docs = 0

        query_ids = []
        query_id_group = []
        doc_ids = []
        doc_id_group = []
        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])

                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])

                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))


                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        query_ids.append(self.data_rows[i][QUESTION_ID_INDEX])
                        doc_id_group.append(doc_ids)
                    else:
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []
                    doc_ids = []
            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])


                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    doc_id_group.append(doc_ids)
                    query_ids.append(self.data_rows[i][QUESTION_ID_INDEX])
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0
        return queries, docs, labels, query_ids, doc_id_group



if __name__ == '__main__':

    q_iterable = MyWikiIterable('query', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))
    d_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))
    l_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))    

    q_lens, doc_lens, d_lens = [], [], []
    for q, doc in zip(q_iterable, d_iterable):
        q_lens.append(len(q))
        doc_lens.append(len(doc))
        for d in doc:
            d_lens.append(len(d))

    print('max query, max number of docs per query and max number of docs')
    print(max(q_lens), max(doc_lens), max(d_lens))
    # 33 29 230

    import numpy as np
    q_lens = np.array(q_lens)
    d_lens = np.array(d_lens)
    doc_lens = np.array(doc_lens)
    print('Average query, average number of docs per query and average number of docs')
    print(np.mean(q_lens), np.mean(doc_lens), np.mean(d_lens))
    # 10.405203405865658 5.105676442762536 24.902959215817074

    kv_model = api.load('glove-wiki-gigaword-50')
    model = BiDAF(q_iterable, d_iterable, l_iterable, kv_model, text_maxlen=51, unk_handle_method='zero', epochs=3, batch_size=20)

    # Example of how prediction works
    print('Hello there result: ', model.tiny_predict('Hello there', 'general kenobi'))
    print('Hello there batch: ', model.batch_tiny_predict('Hello there', ['gengeral kenowbi', 'i am groot', 'I dont wear boot']))

    queries, doc_group, label_group, query_ids, doc_id_group = MyOtherWikiIterable(os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-test.tsv')).get_stuff()
    i=0

    with open('jpred', 'w') as f:
        for q, doc, labels, q_id, d_ids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
            batch_score = model.batch_tiny_predict(q, doc)
            for d, l, d_id, bscore in zip(doc, labels, d_ids, batch_score):
                # my_score = str(model.tiny_predict(q,d))
                my_score = bscore[1]
                print(i, my_score)
                i += 1
                f.write(q_id + '\t' + 'Q0' + '\t' + str(d_id) + '\t' + '99' + '\t' + str(my_score) + '\t' + 'STANDARD' + '\n')
    print("Prediction done. Saved as %s" % 'jpred')

    with open('qrels', 'w') as f:
        for q, doc, labels, q_id, d_ids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
            for d, l, d_id in zip(doc, labels, d_ids):
                f.write(q_id + '\t' +  '0' + '\t' +  str(d_id) + '\t' + str(l) + '\n')
    print("qrels done. Saved as %s" % 'qrels')

