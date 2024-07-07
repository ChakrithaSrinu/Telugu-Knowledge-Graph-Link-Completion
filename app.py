import streamlit as st
import pandas as pd
import kgc_final 
from kgc_final import compare_triplets, indexed_triplets, neg_indexed_triplets, true_embeddings_dict, neg_embeddings_dict
import torch
import torch.optim as optim
  # Ensure kgc_final.py is in the same directory or properly importable

# Set up Streamlit app
st.title("Telugu Knowledge Graph Link Completion")

# Upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    csv_content = uploaded_file.read().decode("utf-8")
    
    # Save the uploaded file content to a temporary file
    temp_csv_file_path = "temp_triplets.csv"
    with open(temp_csv_file_path, 'w', encoding='utf-8') as temp_csv_file:
        temp_csv_file.write(csv_content)
    
    # Read the CSV file to use it as input for the model
    triplets = kgc_final.read_triplets(temp_csv_file_path)
    
    entities = set()
    relations = set()

    for head, relation, tail in triplets:
        entities.add(head)
        entities.add(tail)
        relations.add(relation)

    entity2idx = {entity: idx for idx, entity in enumerate(entities)}
    relation2idx = {relation: idx for idx, relation in enumerate(relations)}

    indexed_triplets = [(entity2idx[head], relation2idx[relation], entity2idx[tail]) for head, relation, tail in triplets]

    thead = torch.LongTensor([triplet[0] for triplet in indexed_triplets])
    trelation = torch.LongTensor([triplet[1] for triplet in indexed_triplets])
    ttail = torch.LongTensor([triplet[2] for triplet in indexed_triplets])

    # Generate negative triplets
    negative_triplets = kgc_final.generate_negative_triplets(triplets, entities)

    neg_entities = set()
    for head, relation, tail in negative_triplets:
        neg_entities.add(head)
        neg_entities.add(tail)

    neg_indexed_triplets = [(entity2idx[head], relation2idx[relation], entity2idx[tail]) for head, relation, tail in negative_triplets]

    neg_head = torch.LongTensor([triplet[0] for triplet in neg_indexed_triplets])
    neg_relation = torch.LongTensor([triplet[1] for triplet in neg_indexed_triplets])
    neg_tail = torch.LongTensor([triplet[2] for triplet in neg_indexed_triplets])

    # Load and train the model
    num_entities = len(entity2idx)
    num_relations = len(relation2idx)
    embedding_dim = 64
    margin = 1.0
    learning_rate = 0.001
    num_epochs = 100

    model = kgc_final.TransEModel(num_entities, num_relations, embedding_dim, margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        pos_dist, neg_dist_head, neg_dist_tail = model(thead, trelation, ttail, neg_head, neg_tail)
        loss = model.loss(pos_dist, neg_dist_head, neg_dist_tail)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    entity_embeddings = model.entity_embeddings.weight.data.cpu().numpy()
    relation_embeddings = model.relation_embeddings.weight.data.cpu().numpy()

    true_embeddings_dict = {idx: entity_embeddings[idx] for entity, idx in entity2idx.items()}
    true_embeddings_dict.update({idx: relation_embeddings[idx] for relation, idx in relation2idx.items()})

    neg_embeddings_dict = true_embeddings_dict.copy()

    hidden_rels = kgc_final.compare_triplets(indexed_triplets, neg_indexed_triplets, true_embeddings_dict, neg_embeddings_dict, threshold=0.4)
    hidden_relations = list(set(hidden_rels))

    valid_neg_tuples = []
    entity_keys = list(entity2idx.keys())
    entity_values = list(entity2idx.values())
    relation_keys = list(relation2idx.keys())
    relation_values = list(relation2idx.values())

    for triplet in hidden_relations:
        head_entity = entity_keys[entity_values.index(triplet[0])]
        relation = relation_keys[relation_values.index(triplet[1])]
        tail_entity = entity_keys[entity_values.index(triplet[2])]
        valid_neg_tuples.append((head_entity, relation, tail_entity))

    df = pd.DataFrame(valid_neg_tuples, columns=["head", "relation", "tail"])
    st.write("Hidden Relations")
    st.dataframe(df)
