import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from collections import Counter
import nltk
# from nltk.translate import bleu_score
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Step 1: Create Vocabulary Mapping Function
def build_vocab(texts):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
    vocab['<PAD>'] = 0
    vocab['<SOS>'] = len(vocab)
    vocab['<EOS>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)  # Unknown token

    return vocab


# Step 2: Load and preprocess the data
class NYTDataset(Dataset):
    def __init__(self, csv_file, abstract_vocab, title_vocab, title_vocab_inv):
        self.data = pd.read_csv(csv_file).dropna()
        self.abstracts = self.data['abstract'].tolist()
        self.titles = self.data['title'].tolist()
        self.abstract_vocab = abstract_vocab
        self.title_vocab = title_vocab
        self.title_vocab_inv = title_vocab_inv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        abstract = self.abstracts[idx]
        title = self.titles[idx]

        abstract_indices = [self.abstract_vocab.get(word, self.abstract_vocab['<UNK>']) for word in abstract.split()]
        title_indices = [self.title_vocab.get(word, self.title_vocab['<UNK>']) for word in title.split()]

        return abstract_indices, title_indices


def collate_fn(batch):
    # Sort batch by sequence length (descending order) for pack_padded_sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_seqs, target_seqs = zip(*batch)

    # Pad sequences
    input_padded = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in input_seqs], batch_first=True)
    target_padded = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in target_seqs], batch_first=True)

    return input_padded, target_padded


# Step 3: Define the Seq2Seq Model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output.squeeze(0))
        return output, hidden


# Step 4: Training loop
def train(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, dataset, batch_size, num_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for input_seqs, target_seqs in dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_lengths = [len(seq) for seq in input_seqs]

            # Move input sequences and target sequences to the device
            input_seqs = input_seqs.to(device)
            target_seqs = target_seqs.to(device)

            encoder_outputs, encoder_hidden = encoder(input_seqs, input_lengths)

            decoder_input = torch.tensor([[dataset.title_vocab['<SOS>']] * batch_size], device=device)
            decoder_hidden = encoder_hidden

            loss = 0
            target_length = target_seqs.size(1)
            for t in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_output = decoder_output.to(device)  # Move decoder output to device
                loss += criterion(decoder_output, target_seqs[:, t])
                decoder_input = target_seqs[:, t].unsqueeze(0)  # Teacher forcing

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')

    # Save trained models
    torch.save(encoder.state_dict(), 'encoder_model.pth')
    torch.save(decoder.state_dict(), 'decoder_model.pth')


def test(encoder, decoder, dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    predictions = []
    references = []
    hypotheses = []

    with torch.no_grad():
        for input_seqs, target_seqs in dataloader:
            # Calculate input_lengths
            input_lengths = [len(seq) for seq in input_seqs]

            input_seqs = torch.tensor(input_seqs, dtype=torch.long, device=device).clone().detach()

            encoder_outputs, encoder_hidden = encoder(input_seqs, input_lengths)

            decoder_input = torch.tensor([[dataset.title_vocab['<SOS>']]], device=device)
            decoder_hidden = encoder_hidden

            predicted_title_indices = []
            for _ in range(MAX_LENGTH):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                _, top_index = decoder_output.topk(1)
                predicted_title_indices.append(top_index.item())
                if top_index == dataset.title_vocab['<EOS>']:
                    break
                decoder_input = top_index.squeeze(0).unsqueeze(0)

            predicted_title_words = [dataset.title_vocab_inv[idx] for idx in predicted_title_indices]
            predicted_title = ' '.join(predicted_title_words)
            predictions.append(predicted_title)

            target_title_words = [dataset.title_vocab_inv[idx.item()] for idx in target_seqs[0] if
                                  idx.item() != dataset.title_vocab['<PAD>']]
            reference_title = ' '.join(target_title_words)
            references.append(reference_title)
            hypotheses.append(predicted_title)

    # Compute BLEU score
    bleu_score = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses])

    # Compute ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, references, avg=True)

    print(f'BLEU Score: {bleu_score}')
    print(f'ROUGE Scores: {rouge_scores}')

    return predictions


# Example usage:
if __name__ == '__main__':
    # nltk.download('punkt')  # Download NLTK tokenizer data (if not already downloaded)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load dataset
    dataset_file = 'training_data.csv'
    data = pd.read_csv(dataset_file).dropna()

    # Build vocabularies
    abstract_vocab = build_vocab(data['abstract'].tolist())
    title_vocab = build_vocab(data['title'].tolist())
    title_vocab_inv = {v: k for k, v in title_vocab.items()}

    # Define model parameters
    input_size = len(abstract_vocab)
    output_size = len(title_vocab)
    hidden_size = 256
    batch_size = 16
    num_epochs = 150
    MAX_LENGTH = 20

    # Initialize model components
    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = Decoder(hidden_size, output_size).to(device)

    # Load saved state dictionaries into the models
    encoder.load_state_dict(torch.load('encoder_model.pth'))
    decoder.load_state_dict(torch.load('decoder_model.pth'))

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Prepare dataset and train the model
    train_dataset = NYTDataset(dataset_file, abstract_vocab, title_vocab, title_vocab_inv)
    train(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, train_dataset, batch_size, num_epochs)

    # Load testing dataset
    test_dataset_file = 'testing_data.csv'
    test_data = pd.read_csv(test_dataset_file).dropna()

    # Prepare testing dataset
    test_dataset = NYTDataset(test_dataset_file, abstract_vocab, title_vocab, title_vocab_inv)

    # Test the trained model on the testing dataset
    predictions = test(encoder, decoder, test_dataset, device)
    # print(predictions)

    # Example: Generate title for a given abstract using the trained model
    example_abstract = "Britain’s Parliament passed contentious legislation to allow the deportation of asylum seekers to Africa, a political victory for the Prime Minister."
    example_abstract_indices = [abstract_vocab.get(word, abstract_vocab['<UNK>']) for word in example_abstract.split()]
    example_abstract_tensor = torch.tensor(example_abstract_indices, dtype=torch.long, device=device).unsqueeze(0)

    encoder.eval()
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(example_abstract_tensor, [len(example_abstract_tensor)])

        decoder_input = torch.tensor([[title_vocab['<SOS>']]], device=device)
        decoder_hidden = encoder_hidden

        generated_title_indices = []
        for _ in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, top_index = decoder_output.topk(1)
            generated_title_indices.append(top_index.item())
            if top_index == title_vocab['<EOS>']:
                break
            decoder_input = top_index.squeeze(0).unsqueeze(0)

    generated_title_words = [title_vocab_inv[idx] for idx in generated_title_indices]
    generated_title = ' '.join(generated_title_words)

    print("Generated Title:", generated_title)
