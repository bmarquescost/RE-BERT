import nltk 
nltk.download('stopwords')

import random
import numpy as np
import networkx as nx
import argparse

APPS = ['eBay', 'WhatsApp', 'Facebook', 'Evernote', 'Twitter', 'Netflix', 'PhotoEditor', 'Spotify']
                
def get_sequence_to_token(lines, TOKEN='$T$'):
    sequence_to_tokens = {}
    for line_idx in range(0, len(lines), 3):
        sentence = lines[line_idx].strip()
        token = lines[line_idx + 1].strip()
        iob_class = int(lines[line_idx + 2])

        full_sentence = sentence.replace(TOKEN, token).lower()

        if full_sentence in sequence_to_tokens:
            sequence_to_tokens[full_sentence].append((token, iob_class))
        else:
            sequence_to_tokens[full_sentence] = [(token, iob_class)]
    
    return sequence_to_tokens

def negative_sampling(Graph, sequence_to_tokens, n_positive_req):
    negatives = []
    stop_words = nltk.corpus.stopwords.words('english')
    for sentence, tokens_info in sequence_to_tokens.items():
        for token, iob_class in tokens_info:
            if iob_class == -1 and token not in stop_words:
                negatives.append((token, sentence))

    ns = random.sample(negatives, n_positive_req)
    for token, sentence in ns:
        Graph.add_node(token, iob_class=-1)
        Graph.add_edge(sentence, token)

def populate_train_data(Graph, app_name, file_path='./dataset_train',):
    TOKEN = "$T$"

    lines = []
    with open(f"{file_path}/train_{app_name}.txt" , 'r') as f:
        lines = f.readlines()

    sequence_to_tokens = get_sequence_to_token(lines, TOKEN)
    
    positive_req = 0
    for sentence, tokens in sequence_to_tokens.items():
        Graph.add_node(sentence)
        for token, iob_class in tokens:
            if iob_class != -1:
                positive_req += 1
                Graph.add_node(token, iob_class=1)
                Graph.add_edge(sentence, token)
    
    negative_sampling(Graph, sequence_to_tokens, positive_req)

def populate_test_layer(Graph, app_name, path_test, path_model_pred):
    APPS.remove(app_name)
    
    for app in APPS:
        with open(f"{path_model_pred}/{app_name}/{app_name}_model_on_{app}.txt") as model_pred_fp, open(f"{path_test}/test_data_{app}.txt") as test_fp:
            test_sentences = test_fp.readlines()    
            model_preds    = model_pred_fp.readlines()
        for sentence, extracted_data in zip(test_sentences, model_preds):
            Graph.add_node(sentence.strip())

            extracted_requirements = [req.strip() for req in extracted_data.split(',')[0].split(';')]
            Graph.add_nodes_from(extracted_requirements, iob_class=1)
            for req in extracted_requirements: Graph.add_edge(sentence.strip().lower(), req)
    
    APPS.append(app_name)

def create_graph(opt):
    G = nx.Graph()
    # Create train dataset layer + negative sampling
    populate_train_data(G, opt.dataset, opt.train_folder)

    # Create test dataset layer
    populate_test_layer(G, opt.dataset, opt.test_folder, opt.models_pred_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', default='./dataset_train', type=str)
    parser.add_argument('--test_folder', default='./datasets_iob', type=str)
    parser.add_argument('--models_pred_folder', default='./models_predictions', type=str)

    # To be changed
    parser.add_argument('--dataset', type=str, help='app_name')

    opt = parser.parse_args()

    create_graph(opt)

if __name__ == '__main__':
    main()