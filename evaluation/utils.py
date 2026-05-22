"""
Utility Functions for Argument Extraction Evaluation

Helper functions for data loading, validation, and common operations.

Author: Automatic Evaluation Framework
License: MIT
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate evaluation data."""
    
    @staticmethod
    def load_gold_annotations(file_path: str) -> List[Dict]:
        """
        Load gold annotations from JSON file.
        
        Expected format:
        [
            {
                "id": 1,
                "text": "...",
                "intent": "...",
                "arguments": {
                    "participants": [...],
                    "time": [...],
                    "location": [...],
                    "topic": [...]
                }
            },
            ...
        ]
        
        Args:
            file_path: Path to gold annotations JSON
            
        Returns:
            List of annotated emails
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Gold annotations must be a JSON list")
            
            logger.info(f"Loaded {len(data)} gold annotations from {file_path}")
            return data
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Gold annotations file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {str(e)}", 
                                      doc=file_path, pos=0)
    
    @staticmethod
    def load_predictions(file_path: str) -> Dict[int, Dict]:
        """
        Load model predictions from JSON file.
        
        Expected format:
        {
            "1": {
                "participants": [...],
                "time": [...],
                "location": [...],
                "topic": [...]
            },
            ...
        }
        
        Args:
            file_path: Path to predictions JSON
            
        Returns:
            Dictionary mapping email IDs to predictions
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string keys to integers if necessary
            predictions = {}
            for key, value in data.items():
                try:
                    int_key = int(key)
                    predictions[int_key] = value
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid key: {key}")
            
            logger.info(f"Loaded {len(predictions)} predictions from {file_path}")
            return predictions
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Predictions file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {str(e)}", 
                                      doc=file_path, pos=0)
    
    @staticmethod
    def merge_gold_and_predictions(
        gold_annotations: List[Dict],
        predictions: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Merge gold annotations with model predictions.
        
        Args:
            gold_annotations: Gold annotations list
            predictions: Predictions dictionary
            
        Returns:
            List of merged email data
        """
        merged = []
        
        for annotation in gold_annotations:
            email_id = annotation.get("id")
            
            # Get predictions for this email
            pred = predictions.get(email_id, {})
            
            # Skip if no predictions
            if not pred:
                logger.warning(f"No predictions found for email {email_id}")
                continue
            
            # Merge
            merged_item = {
                "id": email_id,
                "text": annotation.get("text", ""),
                "intent": annotation.get("intent", ""),
                "arguments": annotation.get("arguments", {}),
                "predicted": pred
            }
            
            merged.append(merged_item)
        
        logger.info(f"Merged {len(merged)} emails")
        return merged


class DataValidator:
    """Validate evaluation data structure."""
    
    @staticmethod
    def validate_gold_annotations(data: List[Dict]) -> bool:
        """
        Validate gold annotations structure.
        
        Args:
            data: Gold annotations data
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If structure is invalid
        """
        required_fields = {"id", "text", "intent", "arguments"}
        required_arg_types = {"participants", "time", "location", "topic"}
        
        for i, annotation in enumerate(data):
            # Check required fields
            if not isinstance(annotation, dict):
                raise ValueError(f"Annotation {i} is not a dictionary")
            
            missing = required_fields - set(annotation.keys())
            if missing:
                raise ValueError(f"Annotation {i} missing fields: {missing}")
            
            # Validate arguments structure
            arguments = annotation.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError(f"Annotation {i}: arguments is not a dict")
            
            missing_args = required_arg_types - set(arguments.keys())
            if missing_args:
                raise ValueError(f"Annotation {i} missing argument types: {missing_args}")
            
            # Validate argument values are lists
            for arg_type, values in arguments.items():
                if not isinstance(values, list):
                    raise ValueError(
                        f"Annotation {i}: {arg_type} should be a list, "
                        f"got {type(values).__name__}"
                    )
        
        return True
    
    @staticmethod
    def validate_predictions(data: Dict) -> bool:
        """
        Validate predictions structure.
        
        Args:
            data: Predictions data
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If structure is invalid
        """
        required_arg_types = {"participants", "time", "location", "topic"}
        
        for email_id, predictions in data.items():
            if not isinstance(predictions, dict):
                raise ValueError(f"Predictions for email {email_id} is not a dict")
            
            missing_args = required_arg_types - set(predictions.keys())
            if missing_args:
                raise ValueError(
                    f"Email {email_id} predictions missing argument types: {missing_args}"
                )
            
            # Validate argument values are lists
            for arg_type, values in predictions.items():
                if not isinstance(values, list):
                    raise ValueError(
                        f"Email {email_id}: {arg_type} should be a list, "
                        f"got {type(values).__name__}"
                    )
        
        return True


class DataPreprocessor:
    """Preprocess evaluation data."""
    
    @staticmethod
    def clean_argument_text(text: str) -> str:
        """
        Clean argument text.
        
        Operations:
        - Trim whitespace
        - Normalize Unicode
        - Remove extra spaces
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    @staticmethod
    def clean_arguments_list(arguments: List[str]) -> List[str]:
        """
        Clean list of arguments.
        
        Args:
            arguments: List of argument texts
            
        Returns:
            List of cleaned arguments
        """
        cleaned = []
        
        for arg in arguments:
            clean_arg = DataPreprocessor.clean_argument_text(arg)
            
            # Skip empty strings
            if clean_arg:
                cleaned.append(clean_arg)
        
        return cleaned
    
    @staticmethod
    def clean_email_data(email_data: Dict) -> Dict:
        """
        Clean all arguments in email data.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            Cleaned email data
        """
        cleaned = email_data.copy()
        
        # Clean arguments
        if "arguments" in cleaned:
            for arg_type in ["participants", "time", "location", "topic"]:
                if arg_type in cleaned["arguments"]:
                    cleaned["arguments"][arg_type] = DataPreprocessor.clean_arguments_list(
                        cleaned["arguments"][arg_type]
                    )
        
        # Clean predicted arguments
        if "predicted" in cleaned:
            for arg_type in ["participants", "time", "location", "topic"]:
                if arg_type in cleaned["predicted"]:
                    cleaned["predicted"][arg_type] = DataPreprocessor.clean_arguments_list(
                        cleaned["predicted"][arg_type]
                    )
        
        return cleaned


class EvaluationUtils:
    """General utility functions."""
    
    @staticmethod
    def create_output_directory(output_dir: str) -> Path:
        """
        Create output directory if it doesn't exist.
        
        Args:
            output_dir: Directory path
            
        Returns:
            Path object
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {path}")
        return path
    
    @staticmethod
    def get_timestamp_str() -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def format_metric(value: float, precision: int = 4) -> str:
        """
        Format metric value.
        
        Args:
            value: Metric value
            precision: Decimal places
            
        Returns:
            Formatted string
        """
        return f"{value:.{precision}f}"
    
    @staticmethod
    def get_argument_type_color(argument_type: str) -> str:
        """
        Get color for argument type (for visualizations).
        
        Args:
            argument_type: Type of argument
            
        Returns:
            Color hex code
        """
        colors = {
            "participants": "#FF6B6B",
            "time": "#4ECDC4",
            "location": "#45B7D1",
            "topic": "#FFA07A"
        }
        return colors.get(argument_type, "#808080")
    
    @staticmethod
    def flatten_dict(nested_dict: Dict, parent_key: str = '') -> Dict:
        """
        Flatten nested dictionary.
        
        Args:
            nested_dict: Nested dictionary
            parent_key: Parent key prefix
            
        Returns:
            Flattened dictionary
        """
        items = []
        
        for k, v in nested_dict.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(EvaluationUtils.flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
