from socket import create_connection
import nltk 
nltk.download('stopwords')

import random
import numpy as np
import networkx as nx
import argparse

APPS = ['eBay', 'WhatsApp', 'Facebook', 'Evernote', 'Twitter', 'Netflix', 'PhotoEditor', 'Spotify']
                
def get_sequence_to_token(sequence_to_tokens, lines, TOKEN='$T$'):
    for line_idx in range(0, len(lines), 3):
        sentence = lines[line_idx].strip().lower()
        token = lines[line_idx + 1].strip().lower()
        iob_class = int(lines[line_idx + 2])

        full_sentence = sentence.replace(TOKEN, token).lower()

        if full_sentence in sequence_to_tokens:
            sequence_to_tokens[full_sentence].append((token, iob_class))
        else:
            sequence_to_tokens[full_sentence] = [(token, iob_class)]
    

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

def populate_train_data(Graph, file_path='./dataset_train',):
    TOKEN = "$T$"

    sequence_to_tokens = {}
    for app in APPS:
        lines = []
        with open(f"{file_path}/train_{app}.txt" , 'r') as f:
            lines = f.readlines()

        get_sequence_to_token(sequence_to_tokens, lines, TOKEN)
        
    positive_req = 0
    for sentence, tokens in sequence_to_tokens.items():
        Graph.add_node(sentence)
        for token, iob_class in tokens:
            if iob_class != -1:
                positive_req += 1
                Graph.add_node(token, iob_class=1)
                Graph.add_edge(sentence, token)
    
    negative_sampling(Graph, sequence_to_tokens, positive_req)

def populate_test_layer(Graph, path_test, path_model_pred):
    for model_app in APPS:
        print(APPS)
        APPS.remove(model_app)
        Graph.add_node(model_app)
        
        for app in APPS:
            print(f"Modelo: {path_model_pred}/{model_app}/{model_app}_model_on_{app}.txt")
            print(f"{path_test}/test_data_{app}.txt")
            with open(f"{path_model_pred}/{model_app}/{model_app}_model_on_{app}.txt") as model_pred_fp, open(f"{path_test}/test_data_{app}.txt") as test_fp:
                test_sentences = test_fp.readlines()    
                model_preds    = model_pred_fp.readlines()

            for sentence, extracted_data in zip(test_sentences, model_preds):
                Graph.add_node(sentence.strip().lower())
                tmp = extracted_data.split(',')[0].split(';')
                extracted_requirements = [req.strip().lower() for req in tmp]
                # Graph.add_nodes_from(extracted_requirements, iob_class=1)
                # print(f"{sentence.strip()}: {extracted_requirements}")
                for req in extracted_requirements:
                    # print(req)
                    Graph.add_node(req, iob_class = 1)
                    Graph.add_edge(sentence.strip().lower(), req)
                    Graph.add_edge(model_app, req)
        APPS.insert(0, model_app)

def create_graph(opt):
    G = nx.Graph()
    # Create train dataset layer + negative sampling
    populate_train_data(G, opt.train_folder)

    # Create test dataset layer
    populate_test_layer(G, opt.test_folder, opt.models_pred_folder)

    return G

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', default='./dataset_train', type=str)
    parser.add_argument('--test_folder', default='./datasets_iob', type=str)
    parser.add_argument('--models_pred_folder', default='./models_predictions', type=str)

    opt = parser.parse_args()

    graph = create_graph(opt)
    
    
if __name__ == '__main__':
    main()