import nltk 
import random
import numpy as np
import networkx as nx

APPS = ['eBay', 'WhatsApp', 'Facebook', 'Evernote', 'Twitter', 'Netflix', 'PhotoEditor', 'Spotify']
                
def get_sequence_to_token(lines, TOKEN='$T$'):
    sequence_to_tokens = {}
    for line_idx in range(0, len(lines), 3):
        sentence = lines[line_idx].strip()
        token = lines[line_idx + 1].strip()
        iob_class = int(lines[line_idx + 2])

        full_sentence = sentence.replace(TOKEN, token)

        if full_sentence in sequence_to_tokens:
            sequence_to_tokens[full_sentence].append((token, iob_class))
        else:
            sequence_to_tokens[full_sentence] = [(token, iob_class)]
    
    return sequence_to_tokens

def negative_sampling(Graph, sequence_to_tokens, n_positive_req):
    negatives = []
    for sentence, tokens_info in sequence_to_tokens.items():
        for token, iob_class in tokens_info:
            if iob_class == -1:
                negatives.append((token, sentence))
    
    ns = random.sample(negatives, n_positive_req)
    for token, sentence in ns:
        Graph.add_node(token, iob_class=-1)
        Graph.add_edge(sentence, token)

def populate_train_layer(Graph, app_name, file_path='./dataset_train',):
    if app_name not in APPS: return None
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

def populate_test_layer(Graph, path_test, app_name, path_model_pred):
    APPS.remove(app_name)
    
    for app in APPS:
        with open(f"{path_model_pred}/{app_name}/{app_name}_model_on_{app}.txt") as model_pred_fp, open(f"{path_test}/test_data_{app}.txt") as test_fp:
            test_sentences = test_fp.readlines()    
            model_preds    = model_pred_fp.readlines()
        for sentence, extracted_data in zip(test_sentences, model_preds):
            Graph.add_node(sentence.strip())

            extracted_requirements = [req.strip() for req in extracted_data.split(',')[0].split(';')]
            Graph.add_nodes_from(extracted_requirements, iob_class=1)
            for req in extracted_requirements: Graph.add_edge(sentence.strip(), req)
    
    APPS.append(app_name)

def create_graph(train_file_path, test_file_path, model_pred_path, app_name):
    G = nx.Graph()
    # Create train dataset layer (upper layers nodes) + negative sampling
    populate_train_layer(G, app_name, train_file_path)

    # Create test dataset layer
    populate_test_layer(G, test_file_path, app_name, model_pred_path)

def main():
    create_graph('./dataset_train', './datasets_iob' , './models_predictions')

if __name__ == '__main__':
    main()