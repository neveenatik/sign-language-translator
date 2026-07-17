"""
Gesture recognition module for ASL translation.
Uses a neural network to classify hand gestures into sign language letters/words.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, List
import os


class GestureRecognizer:
    """
    Recognizes ASL gestures from hand landmarks.
    Supports both inference with pre-trained models and training new models.
    """

    # ASL alphabet
    ASL_LABELS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'delete'
    ]

    def __init__(self, model_path: Optional[str] = None, labels: Optional[List[str]] = None):
        """
        Initialize the gesture recognizer.
        
        Args:
            model_path: Path to pre-trained model. If None, creates a new model.
            labels: Ordered class labels to recognize. Defaults to the ASL
                alphabet for backward compatibility.
        """
        self.labels = labels if labels is not None else list(self.ASL_LABELS)
        self.model = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Build a neural network for gesture classification.
        
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(63,)),  # 21 landmarks × 3 coords
            
            # Dense layers with dropout
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer
            tf.keras.layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def predict(self, landmarks: np.ndarray, confidence_threshold: float = 0.6) -> Optional[Dict]:
        """
        Predict gesture from hand landmarks.
        
        Args:
            landmarks: Normalized hand landmarks (63,) array
            confidence_threshold: Minimum confidence to return prediction
            
        Returns:
            Dict with 'label' and 'confidence' or None if below threshold
        """
        if self.model is None:
            return None
            
        # Add batch dimension
        landmarks_batch = np.expand_dims(landmarks, axis=0)
        
        # Get predictions
        predictions = self.model.predict(landmarks_batch, verbose=0)[0]
        confidence = np.max(predictions)
        predicted_idx = np.argmax(predictions)
        
        if confidence >= confidence_threshold:
            return {
                'label': self.labels[predicted_idx],
                'confidence': float(confidence),
                'all_predictions': {
                    self.labels[i]: float(predictions[i]) 
                    for i in range(len(self.labels))
                }
            }
        
        return None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        Train the gesture recognition model.
        
        Args:
            X_train: Training landmarks data
            y_train: Training labels (one-hot encoded)
            X_val: Validation landmarks data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.is_trained = True
        return history.history

    def save_model(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a pre-trained model.
        
        Args:
            path: Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(path)
            self.is_trained = True
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_labels(self) -> List[str]:
        """Get list of recognizable labels."""
        return self.labels
