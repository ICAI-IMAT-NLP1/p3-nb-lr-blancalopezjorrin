from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []
    
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            parts: List[str] = line.strip().split('\t')  # Split by tab
            
            if len(parts) < 2:
                continue  # Skip malformed lines
            
            sentence: str = parts[0]
            label: int  = int(parts[-1])
            tokens: List[str] = tokenize(sentence)  # Tokenize sentence
            example: SentimentExample = SentimentExample(tokens, label)
            examples.append(example)
            
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    
    for example in examples:
        for word in example.words:
            if word not in vocab:
                vocab[word] = len(vocab)  # Assign a unique index
    
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # Convert list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab), dtype=torch.float)

    for word in text:
        if word in vocab:
            index = vocab[word]
            if binary:
                bow[index] = 1  # Binary representation
            else:
                bow[index] += 1  # Frequency-based representation
    
    return bow
