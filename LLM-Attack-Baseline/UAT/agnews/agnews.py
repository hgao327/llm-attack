import sys
import os.path
import csv
import re
from sklearn.neighbors import KDTree
import torch
import torch.optim as optim
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
sys.path.append('..')
import utils
import attacks

def simple_tokenize(text):
    """Simple tokenizer without spacy dependency"""
    # Basic cleaning and tokenization
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    tokens = text.split()
    return [Token(t) for t in tokens if t.strip()]

# AG News labels
LABEL_MAP = {
    "1": "World",
    "2": "Sports", 
    "3": "Business",
    "4": "Sci/Tech"
}

# Simple LSTM classifier
class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

def read_agnews_data(file_path, token_indexers):
    """Read AG News CSV file and return list of instances"""
    instances = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                label = row[0]  # "1", "2", "3", or "4"
                title = row[1]
                content = row[2]
                text = title + " " + content  # Combine title and content
                
                tokens = simple_tokenize(text)  # Use simple tokenizer
                if len(tokens) > 0:
                    text_field = TextField(tokens, token_indexers)
                    label_field = LabelField(label)
                    instance = Instance({"tokens": text_field, "label": label_field})
                    instances.append(instance)
    
    return instances

EMBEDDING_TYPE = "w2v"

def main():
    # Dataset paths
    train_path = "/home/liubingshan/datasets/ag_news/train.csv"
    test_path = "/home/liubingshan/datasets/ag_news/test.csv"
    
    # Token indexer
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    token_indexers = {"tokens": single_id_indexer}
    
    print("Loading AG News dataset...")
    train_data = read_agnews_data(train_path, token_indexers)
    test_data = read_agnews_data(test_path, token_indexers)
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Build vocabulary
    vocab = Vocabulary.from_instances(train_data + test_data)
    
    # Load word embeddings
    if EMBEDDING_TYPE == "w2v":
        embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        weight = _read_pretrained_embeddings_file(embedding_path,
                                                  embedding_dim=300,
                                                  vocab=vocab,
                                                  namespace="tokens")
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=300,
                                    weight=weight,
                                    trainable=False)
        word_embedding_dim = 300
    else:
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
        word_embedding_dim = 300

    # Initialize model
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()

    # Model save path
    model_path = "/tmp/agnews_" + EMBEDDING_TYPE + "_model.th"
    vocab_path = "/tmp/agnews_" + EMBEDDING_TYPE + "_vocab"
    
    # Train or load model
    if os.path.isfile(model_path):
        print("Loading pre-trained model...")
        vocab = Vocabulary.from_files(vocab_path)
        model = LstmClassifier(word_embeddings, encoder, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    else:
        print("Training model...")
        iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
        iterator.index_with(vocab)
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_data,
                          validation_dataset=test_data,
                          num_epochs=5,
                          patience=1,
                          cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    
    model.train().cuda()

    # Add gradient hooks
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model)

    # Iterator for attack
    universal_perturb_batch_size = 128
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # Filter dataset to one class (attack this class)
    # "1": World, "2": Sports, "3": Business, "4": Sci/Tech
    dataset_label_filter = "3"  # Attack Business news
    target_label = "2"  # Try to flip to Sports
    
    targeted_data = []
    for instance in train_data + test_data:
        if instance['label'].label == dataset_label_filter:
            targeted_data.append(instance)
    
    print(f"\nTotal samples with label '{dataset_label_filter}' ({LABEL_MAP[dataset_label_filter]}): {len(targeted_data)}")

    # Get accuracy before attack
    utils.get_accuracy(model, targeted_data, vocab, trigger_token_ids=None)
    model.train()

    # Initialize trigger tokens
    num_trigger_tokens = 3
    trigger_token_ids = [vocab.get_token_index("the")] * num_trigger_tokens

    # Attack loop
    print("\nStarting attack...")
    for batch in lazy_groups_of(iterator(targeted_data, num_epochs=5, shuffle=True), group_size=1):
        utils.get_accuracy(model, targeted_data, vocab, trigger_token_ids)
        model.train()

        averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids)

        cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                        embedding_weight,
                                                        trigger_token_ids,
                                                        num_candidates=40,
                                                        increase_loss=True)

        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids)

    # Final accuracy
    utils.get_accuracy(model, targeted_data, vocab, trigger_token_ids)

    # ========== Generate and save adversarial samples ==========
    trigger_words = [vocab.get_token_from_index(idx) for idx in trigger_token_ids]
    trigger_text = ' '.join(trigger_words)
    print("\n" + "="*60)
    print(f"Final Trigger: {trigger_text}")
    print("="*60)

    # Generate adversarial samples for ALL classes (not just one class)
    all_data = train_data + test_data
    
    output_file = "adversarial_samples.txt"
    num_samples = min(1000, len(all_data))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Universal Adversarial Trigger: {trigger_text}\n")
        f.write(f"# Each sample keeps its original true label\n")
        f.write("="*60 + "\n\n")

        # Iterate through ALL samples (all classes)
        for i, instance in enumerate(all_data[:num_samples]):
            original_tokens = [token.text for token in instance['tokens'].tokens]
            original_text = ' '.join(original_tokens)
            
            # Get original label (the true label of this sample)
            original_label = instance['label'].label  # "1", "2", "3", or "4"
            label_name = LABEL_MAP.get(original_label, original_label)
            
            adversarial_text = trigger_text + ' ' + original_text

            f.write(f"Sample {i+1}:\n")
            f.write(f"  Original: {original_text[:200]}...\n")  # Truncate long text
            f.write(f"  Adversarial: {trigger_text} {original_text[:200]}...\n")
            f.write(f"  Label: {label_name}\n")
            f.write("\n")

    print(f"\nAdversarial samples saved to: {output_file}")
    print(f"Total samples generated: {num_samples}")

if __name__ == '__main__':
    main()

