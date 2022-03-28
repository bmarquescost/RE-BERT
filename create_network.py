import networkx as nx
import nltk 

def create_graph(sequence_to_tokens):
    G = nx.Graph(layer="train")

    for sentence, tokens in sequence_to_tokens.items():
        G.add_node(sentence)
        for token, iob_class in tokens:
            G.add_node(token, iob_class=iob_class)
            G.add_edge(sentence, token)
    
    for n in G.nodes:
        print(n)
    for e in G.edges:
        print(e)

# Create network for train dataset (upper layers nodes)
def main():
    # File to used to create the network
    FILE_PATH = "./dataset_train/train_WhatsApp.txt"
    TOKEN = "$T$"

    lines = []
    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()

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
    
    create_graph(sequence_to_tokens)

if __name__ == '__main__':
    main()