import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse

from CSE256_PA2_FA24.PA2_code.alibiattention import AliBiTransformerEncoder
from CSE256_PA2_FA24.PA2_code.decoder import TransformerDecoder, pretrain_decoder
from CSE256_PA2_FA24.feedforwardclassifier import FeedForwardClassifier
from CSE256_PA2_FA24.PA2_code.transformer import TransformerEncoder
from CSE256_PA2_FA24.PA2_code.utilities import Utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
# n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_embd = n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def get_max_sequence_length(data_loader):
    max_len = 0
    for batch, _ in data_loader:
        current_max = batch.size(1)
        max_len = max(max_len, current_max)
    return max_len
def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    total_loss = 0
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        logits, _ = decoderLMmodel(X)  # Decoder generates logits for the next token

        # Compute the loss (cross-entropy between logits and target)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), Y.view(-1))

        # loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    # decoderLMmodel.train() no need b/c training in loop
    return perplexity

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run based on question part')
    parser.add_argument('--part', type=str, required=True, help='Question part (1,2,3) to run')

    # Parse the command-line arguments
    args = parser.parse_args()


    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    max_len = get_max_sequence_length(train_CLS_loader)
    print(f"Maximum sequence length in dataset: {max_len}")

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    max_len = get_max_sequence_length(train_LM_loader)
    print(f"Maximum sequence length in dataset: {max_len}")

    obamafile = "speechesdataset/test_LM_obama.txt"
    with open(obamafile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    test_LM_obama = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    test_LM_obama_loader = DataLoader(test_LM_obama, batch_size=batch_size, shuffle=True)


    wbushfile = "speechesdataset/test_LM_wbush.txt"
    with open(wbushfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    test_LM_wbush = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    test_LM_wbush_loader = DataLoader(test_LM_wbush, batch_size=batch_size, shuffle=True)

    ghbushfile = "speechesdataset/test_LM_hbush.txt"
    with open(ghbushfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    test_LM_ghbush = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    test_LM_ghbush_loader = DataLoader(test_LM_ghbush, batch_size=batch_size, shuffle=True)

    if args.part == "part1":
         # for the classification  task, you will train for a fixed number of epochs like this:

        # Then use this when creating your encoder:
        encoder = TransformerEncoder(
            n_layers=n_layer,
            n_heads=n_head,
            n_dim=n_embd,
            hidden_dim=n_hidden,
            vocab_size=tokenizer.vocab_size,
            max_seq_length=block_size
        ).to(device)


        classifier = FeedForwardClassifier(
            n_input=n_embd,  # 64 from main.py
            hidden_dim=n_hidden,  # 100 from main.py
            n_output=n_output,  # 3 from main.py
            encoder=encoder  # Pass the encoder instance
        ).to(device)

        utils = Utilities(tokenizer, encoder)
        utils.sanity_check("This is a much longer, and perhaps unnecessarily verbose test sentence, designed to show the capabilities of the attention mechanism", block_size=32)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()  # For classifier training

        testAccuracies = []
        trainAccuracies = []
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        for epoch in range(epochs_CLS):
            for batch, labels in train_CLS_loader:
                batch = batch.to(device)
                labels = labels.to(device)

                # Get embeddings from transformer encoder
                # embeddings, _ = encoder(batch)

                # Forward pass through classifier
                logits = classifier(batch)

                # Compute loss
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print and save classifier accuracy every epoch
            train_accuracy = compute_classifier_accuracy(classifier, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
            trainAccuracies.append(train_accuracy)
            testAccuracies.append(test_accuracy)
            print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {loss.item()}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%")

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(trainAccuracies, label='train accuracy')
        plt.plot(testAccuracies, label='test accuracy')
        plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid()
        plt.show()

    if args.part == "part2":
        decoder = TransformerDecoder(
            n_layers=4,
            n_heads=2,
            n_dim=64,
            hidden_dim=100,
            vocab_size=tokenizer.vocab_size
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        # pretrained_decoder = pretrain_decoder(decoder, train_LM_loader, criterion, optimizer, device)

        utils = Utilities(tokenizer, decoder)
        # utils.sanity_check("This is a test sentence", block_size=32)

        training_perplexities = []
        obama_perplexities = []
        wbush_perplexities = []
        ghbush_perplexities = []

        # Training loop
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break

            xb, yb = xb.to(device), yb.to(device)

            # Perform forward pass
            logits, _ = decoder(xb)  # Decoder generates logits for the next token

            # Compute the loss (cross-entropy between logits and target)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training perplexity every `eval_interval` iterations
            if i % eval_interval == 0:
                training_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
                training_perplexities.append(training_perplexity)
                print(f"Training Iteration {i}/{max_iters} - Training Perplexity: {training_perplexity:.4f}")

            # Compute and store perplexities for test sets every `eval_interval` iterations
            if i % eval_interval == 0:
                # Obama perplexity
                obama_perplexity = compute_perplexity(decoder, test_LM_obama_loader, eval_iters)
                obama_perplexities.append(obama_perplexity)
                print(f"Testing Iteration {i}/{max_iters} - Obama Perplexity: {obama_perplexity:.4f}")

                # W. Bush perplexity
                wbush_perplexity = compute_perplexity(decoder, test_LM_wbush_loader, eval_iters)
                wbush_perplexities.append(wbush_perplexity)
                print(f"Testing Iteration {i}/{max_iters} - W. Bush Perplexity: {wbush_perplexity:.4f}")

                # G. Bush perplexity
                ghbush_perplexity = compute_perplexity(decoder, test_LM_ghbush_loader, eval_iters)
                ghbush_perplexities.append(ghbush_perplexity)
                print(f"Testing Iteration {i}/{max_iters} - G. Bush Perplexity: {ghbush_perplexity:.4f}")

        # Plot the perplexities for training and test sets
        plt.figure(figsize=(10, 6))  # Adjust the figure size for clarity

        # Plot training perplexity
        plt.plot(training_perplexities, label='Training Perplexity')

        # Plot test set perplexities
        plt.plot(obama_perplexities, label='Obama Test Perplexity')
        plt.plot(wbush_perplexities, label='W. Bush Test Perplexity')
        plt.plot(ghbush_perplexities, label='H. Bush Test Perplexity')

        # Set the title and labels
        plt.title('Perplexity over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Perplexity')

        # Adjust the x-axis to show ticks every 100 iterations
        plt.xticks(range(0, len(training_perplexities), 100))

        # Optionally scale the y-axis for clarity
        plt.ylim(0, 1000)  # Uncomment and adjust if you want to scale y-axis

        # Show grid, legend, and plot
        plt.legend()
        plt.grid()
        plt.show()

        # Training perplexities
        print(f"Training Perplexities by 100 iterations: {training_perplexities}")
        # Final test perplexities
        print(f"Final Obama Test Perplexity: {obama_perplexities[-1]:.2f}")
        print(f"Final W. Bush Test Perplexity: {wbush_perplexities[-1]:.2f}")
        print(f"Final G. Bush Test Perplexity: {ghbush_perplexities[-1]:.2f}")

        print(f"Number of parameters in the decoder: {sum(p.numel() for p in decoder.parameters())}")

    if args.part == "part3":

        # alibi encoder:
        alibiEncoder = AliBiTransformerEncoder(
            n_layers=n_layer,
            n_heads=n_head,
            n_dim=n_embd,
            hidden_dim=n_hidden,
            vocab_size=tokenizer.vocab_size,
            # max_seq_length=block_size
        ).to(device)


        classifier = FeedForwardClassifier(
            n_input=n_embd,  # 64 from main.py
            hidden_dim=n_hidden,  # 100 from main.py
            n_output=n_output,  # 3 from main.py
            encoder=alibiEncoder  # Pass the encoder instance
        ).to(device)

        utils = Utilities(tokenizer, alibiEncoder)
        utils.sanity_check("This is a test sentence", block_size=32)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()  # For classifier training

        testAccuracies = []
        trainAccuracies = []
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        for epoch in range(epochs_CLS):
            for batch, labels in train_CLS_loader:
                batch = batch.to(device)
                labels = labels.to(device)

                # Get embeddings from transformer encoder
                # embeddings, _ = encoder(batch)

                # Forward pass through classifier
                logits = classifier(batch)

                # Compute loss
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print and save classifier accuracy every epoch
            train_accuracy = compute_classifier_accuracy(classifier, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
            trainAccuracies.append(train_accuracy)
            testAccuracies.append(test_accuracy)
            print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {loss.item()}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%")

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(trainAccuracies, label='train accuracy')
        plt.plot(testAccuracies, label='test accuracy')
        plt.title('Training Accuracy over Epochs using AliBi Encodings')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()