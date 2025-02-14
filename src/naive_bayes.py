import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        
        # Compute class prior probabilities P(class)
        self.class_priors = self.estimate_class_priors(labels)
        
        # Store vocabulary size (number of features)
        self.vocab_size = features.shape[1] 
        
        # Compute conditional probabilities P(word | class) with Laplace smoothing
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # Count occurrences of each class
        class_counts: torch.Tensor = torch.bincount(labels.to(torch.int64))  # Count occurrences of each class
        
        # Total number of samples
        total_samples: int = labels.shape[0]  # Total number of samples
        
        # Compute prior probabilities by dividing class counts by the total number of samples
        class_priors: Dict[int, torch.Tensor] = {i: class_counts[i].float() / total_samples for i in range(len(class_counts))}
        
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # Determine the number of classes
        num_classes: int = int(torch.max(labels).item()) + 1  # Determine number of classes
        
        # Store word probability distributions for each class
        class_word_counts: Dict[int, torch.Tensor] = {}
        
        for cls in range(num_classes):
            class_mask: torch.Tensor = (labels == cls)  # Mask for current class
            class_features: torch.Tensor = features[class_mask]  # Extract samples of class
            word_counts: torch.Tensor = torch.sum(class_features, dim=0) + delta  # Apply Laplace smoothing
            total_words: torch.Tensor = torch.sum(word_counts)  # Total count of words in class
            class_word_counts[cls] = word_counts / total_words  # Normalize probabilities
        
        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        
        # Compute log of class priors
        log_priors: torch.Tensor = torch.log(torch.tensor(list(self.class_priors.values())))
        
        # Compute log likelihood for each class (sum of log probabilities of words appearing in the feature)
        log_likelihoods: torch.Tensor = torch.stack([torch.sum(torch.log(self.conditional_probabilities[c]) * feature) for c in self.class_priors.keys()])
        
        # Compute log posterior probabilities: log P(class) + log P(feature | class)
        log_posteriors: torch.Tensor = log_priors + log_likelihoods
        return log_posteriors
    

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # Compute log posteriors and choose the class with the highest probability
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        pred: int = int(torch.argmax(log_posteriors).item())
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # Compute log posterior probabilities and convert them to probabilities using softmax
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.softmax(log_posteriors, dim=0)
        return probs
