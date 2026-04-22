"""
Utility script to extract multimodal features from code diffs for the enhanced transformer model.
This script processes code diffs and extracts various types of features that feed into the 
multimodal embedding system.
"""

import torch
import torch.nn as nn
import ast
import re
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict


class FeatureExtractor:
    """
    Extracts multimodal features from code diffs for the transformer model.
    """
    
    def __init__(self):
        # Define mappings for various features
        self.pattern_mapping = self._create_pattern_mapping()
        self.domain_mapping = self._create_domain_mapping()
        self.change_type_mapping = self._create_change_type_mapping()
        self.security_mapping = self._create_security_mapping()
        
    def _create_pattern_mapping(self):
        """Create mapping for common design patterns and code smells."""
        return {
            'singleton': 0, 'observer': 1, 'factory': 2, 'strategy': 3, 'decorator': 4,
            'adapter': 5, 'facade': 6, 'command': 7, 'template': 8, 'iterator': 9,
            'god_class': 10, 'long_method': 11, 'feature_envy': 12, 'data_class': 13,
            'shotgun_surgery': 14, 'primitive_obsession': 15, 'refused_bequest': 16,
            'cyclomatic_complexity': 17, 'npath_complexity': 18, 'cognitive_complexity': 19
        }
    
    def _create_domain_mapping(self):
        """Create mapping for domain-specific terms."""
        return {
            'authentication': 0, 'payment': 1, 'logging': 2, 'caching': 3, 'validation': 4,
            'serialization': 5, 'deserialization': 6, 'validation': 7, 'authorization': 8,
            'web': 9, 'mobile': 10, 'backend': 11, 'frontend': 12, 'api': 13,
            'healthcare': 14, 'finance': 15, 'e-commerce': 16, 'gaming': 17, 'education': 18
        }
    
    def _create_change_type_mapping(self):
        """Create mapping for change types."""
        return {
            'bug_fix': 0, 'feature_add': 1, 'refactor': 2, 'optimization': 3,
            'documentation': 4, 'test_add': 5, 'dependency_update': 6, 'config_change': 7
        }
    
    def _create_security_mapping(self):
        """Create mapping for security-related patterns."""
        return {
            'sql_injection': 0, 'xss': 1, 'csrf': 2, 'insecure_deserialization': 3,
            'auth_bypass': 4, 'privilege_escalation': 5, 'session_fixation': 6,
            'input_validation': 7, 'output_encoding': 8, 'crypto_weak': 9
        }
    
    def extract_ast_features(self, code: str) -> torch.Tensor:
        """
        Extract AST-based features from code.
        
        Args:
            code: Source code string
            
        Returns:
            AST features tensor
        """
        try:
            tree = ast.parse(code)
            features = []
            
            # Count different node types
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1
            
            # Map common node types to indices
            node_mapping = {
                'FunctionDef': 0, 'ClassDef': 1, 'If': 2, 'For': 3, 'While': 4,
                'Try': 5, 'ExceptHandler': 6, 'With': 7, 'Import': 8, 'ImportFrom': 9,
                'Assign': 10, 'AugAssign': 11, 'AnnAssign': 12, 'Return': 13, 'Yield': 14
            }
            
            # Create feature vector
            for i in range(20):  # 20 different node types
                node_name = [k for k, v in node_mapping.items() if v == i]
                if node_name:
                    features.append(node_counts[node_name[0]])
                else:
                    features.append(0)
            
            # Add control flow features (if-else nesting depth, etc.)
            max_depth = self._calculate_max_nesting_depth(tree)
            features.extend([max_depth, len(node_counts)])  # Add max depth and total nodes
            
            return torch.tensor(features, dtype=torch.float)
        except:
            # Return zeros if parsing fails
            return torch.zeros(22, dtype=torch.float)
    
    def _calculate_max_nesting_depth(self, tree):
        """Calculate maximum nesting depth in the AST."""
        max_depth = 0
        
        def traverse(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                traverse(child, current_depth + 1)
        
        traverse(tree)
        return max_depth
    
    def extract_context_features(self, code: str, surrounding_context: Optional[str] = None) -> torch.Tensor:
        """
        Extract context-based features from code.
        
        Args:
            code: Source code string
            surrounding_context: Context around the code (e.g., file, class, function)
            
        Returns:
            Context features tensor
        """
        features = []
        
        # Simple context features - in a real implementation, this would be more sophisticated
        features.append(1 if 'class' in code else 0)  # Is part of a class
        features.append(1 if 'def' in code else 0)    # Is part of a function
        features.append(1 if 'import' in code else 0) # Has imports
        features.append(1 if 'async' in code else 0)  # Is async code
        
        # Count lines of code
        features.append(len(code.split('\n')))
        
        # Count significant elements
        features.append(len(re.findall(r'def\s+\w+', code)))  # Function definitions
        features.append(len(re.findall(r'class\s+\w+', code)))  # Class definitions
        features.append(len(re.findall(r'if\s+', code)))  # If statements
        features.append(len(re.findall(r'for\s+', code)))  # For loops
        features.append(len(re.findall(r'while\s+', code)))  # While loops
        
        return torch.tensor(features, dtype=torch.float)
    
    def extract_pattern_features(self, code: str) -> torch.Tensor:
        """
        Extract design pattern and code smell features from code.
        
        Args:
            code: Source code string
            
        Returns:
            Pattern features tensor
        """
        features = torch.zeros(23, dtype=torch.float)  # 23 pattern-related features
        
        # Look for common patterns and smells
        code_lower = code.lower()
        
        # Design patterns
        if 'singleton' in code_lower:
            features[self.pattern_mapping['singleton']] = 1
        if 'observer' in code_lower:
            features[self.pattern_mapping['observer']] = 1
        if 'factory' in code_lower:
            features[self.pattern_mapping['factory']] = 1
            
        # Code smells (simplified detection)
        if len(code) > 500:  # Long method heuristic
            features[self.pattern_mapping['long_method']] = 1
            
        # Complexity metrics (simplified)
        if 'if' in code_lower:
            complexity = len(re.findall(r'\bif\s+', code_lower))
            features[self.pattern_mapping['cyclomatic_complexity']] = min(complexity / 10.0, 1.0)
        
        return features
    
    def extract_change_features(self, old_code: str, new_code: str) -> torch.Tensor:
        """
        Extract change type features from code diff.
        
        Args:
            old_code: Original code
            new_code: Modified code
            
        Returns:
            Change type features tensor
        """
        features = torch.zeros(14, dtype=torch.float)  # 14 change-related features
        
        # Determine change type based on diff characteristics
        old_lines = set(old_code.split('\n'))
        new_lines = set(new_code.split('\n'))
        
        added_lines = new_lines - old_lines
        removed_lines = old_lines - new_lines
        
        # Simple heuristics for change type
        if any('bug' in line.lower() or 'fix' in line.lower() for line in added_lines | removed_lines):
            features[self.change_type_mapping['bug_fix']] = 1
        elif any('test' in line.lower() for line in added_lines):
            features[self.change_type_mapping['test_add']] = 1
        elif len(added_lines) > len(removed_lines) * 2:  # Many additions
            features[self.change_type_mapping['feature_add']] = 1
        elif 'refactor' in old_code.lower() or 'refactor' in new_code.lower():
            features[self.change_type_mapping['refactor']] = 1
        elif 'perf' in old_code.lower() or 'perf' in new_code.lower() or 'optim' in old_code.lower() or 'optim' in new_code.lower():
            features[self.change_type_mapping['optimization']] = 1
            
        # Additional continuous features
        features[8] = len(added_lines)  # Number of lines added
        features[9] = len(removed_lines)  # Number of lines removed
        features[10] = len(added_lines) + len(removed_lines)  # Total changes
        features[11] = len(set(re.findall(r'\w+', old_code)))  # Vocabulary in old code
        features[12] = len(set(re.findall(r'\w+', new_code)))  # Vocabulary in new code
        features[13] = abs(len(old_code) - len(new_code)) / max(len(old_code), 1)  # Size change ratio
        
        return features
    
    def extract_security_features(self, code: str) -> torch.Tensor:
        """
        Extract security-related features from code.
        
        Args:
            code: Source code string
            
        Returns:
            Security features tensor
        """
        features = torch.zeros(12, dtype=torch.float)  # 12 security-related features
        
        code_lower = code.lower()
        
        # Look for potential security issues
        if re.search(r"(eval\(|exec\(|os\.system\(|subprocess\.call\(|shellexec)", code):
            features[self.security_mapping['input_validation']] = 1

        if 'sql' in code_lower and ('+' in code or '.format(' in code or f'%{' in code):
            features[self.security_mapping['sql_injection']] = 1

        if 'cookie' in code_lower or 'session' in code_lower:
            features[self.security_mapping['session_fixation']] = 1

        # Look for authentication/authorization patterns
        if any(auth_term in code_lower for auth_term in ['auth', 'login', 'password', 'token', 'jwt', 'oauth']):
            features[10] = 1  # Authentication related
            
        if any(crypto_term in code_lower for crypto_term in ['md5', 'sha1', 'weak', 'insecure']):
            features[self.security_mapping['crypto_weak']] = 1
            
        # Additional security metrics
        features[11] = len(re.findall(r"(?i)(password|secret|key|token)", code)) / max(len(code.split()), 1)
        
        return features
    
    def extract_complexity_features(self, code: str) -> torch.Tensor:
        """
        Extract complexity metrics from code.
        
        Args:
            code: Source code string
            
        Returns:
            Complexity features tensor
        """
        features = torch.zeros(10, dtype=torch.float)  # 10 complexity-related features
        
        try:
            tree = ast.parse(code)
            
            # Cyclomatic complexity (McCabe)
            # Count decision points: if, elif, for, while, except, and, or
            decision_points = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                    decision_points += 1
                elif isinstance(node, ast.BoolOp):  # and, or
                    decision_points += len(node.values) - 1
            
            features[0] = min(decision_points / 10.0, 1.0)  # Normalized cyclomatic complexity
            
            # Lines of code
            features[1] = len(code.split('\n')) / 100.0  # Normalized
            
            # Number of functions/classes
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            features[2] = (len(functions) + len(classes)) / 10.0  # Normalized
            
            # Average parameters per function
            total_params = sum(len(getattr(f, 'args', [])) for f in functions)
            features[3] = total_params / max(len(functions), 1) / 10.0  # Normalized
            
            # Comment density
            comment_lines = sum(1 for line in code.split('\n') if line.strip().startswith('#'))
            features[4] = comment_lines / max(len(code.split('\n')), 1)
            
            # Halstead complexity measures (simplified)
            tokens = re.findall(r'\b\w+\b', code)
            unique_operators = len(set(re.findall(r'(\+|\-|\*|\/|==|!=|<|>|<=|>=|and|or|not|in|is)', code)))
            unique_operands = len(set(token for token in tokens if token not in ['and', 'or', 'not', 'in', 'is']))
            
            features[5] = min(unique_operators / 20.0, 1.0)  # Normalized unique operators
            features[6] = min(unique_operands / 100.0, 1.0)  # Normalized unique operands
            
            # Function length variance
            if functions:
                lengths = [len(f.body) for f in functions]
                features[7] = torch.std(torch.tensor(lengths, dtype=torch.float)).item() if len(lengths) > 1 else 0.0
            
            # Nesting depth
            features[8] = self._calculate_max_nesting_depth(tree) / 10.0  # Normalized
            
            # Identifier length average
            identifiers = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
            avg_id_len = np.mean([len(id) for id in identifiers]) if identifiers else 0
            features[9] = min(avg_id_len / 20.0, 1.0)  # Normalized
            
        except:
            # Return zeros if parsing fails
            pass
        
        return features
    
    def extract_all_features(self, old_code: str, new_code: str, 
                           context: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Extract all multimodal features from code diff.
        
        Args:
            old_code: Original code
            new_code: Modified code
            context: Surrounding context (optional)
            
        Returns:
            Dictionary of feature tensors
        """
        # AST features from new code
        ast_features = self.extract_ast_features(new_code)
        
        # Context features
        context_features = self.extract_context_features(new_code, context)
        
        # Pattern features
        pattern_features = self.extract_pattern_features(new_code)
        
        # Change type features
        change_features = self.extract_change_features(old_code, new_code)
        
        # Security features
        security_features = self.extract_security_features(new_code)
        
        # Complexity features
        complexity_features = self.extract_complexity_features(new_code)
        
        # Create combined feature tensors for the embedding system
        # Each feature tensor needs to have the right shape for the embedding layers
        
        # AST features: [seq_len, 12] where 12 is the number of features
        ast_nodes = torch.cat([ast_features.unsqueeze(0)] * 10, dim=0)  # Repeat for sequence length
        
        # Context features: [seq_len, 4] where 4 is the number of context features
        context_info = torch.cat([context_features.unsqueeze(0)] * 10, dim=0)  # Repeat for sequence length
        
        # Pattern features: [seq_len, 23] 
        patterns = torch.cat([pattern_features.unsqueeze(0)] * 10, dim=0)  # Repeat for sequence length
        
        # Change features: [seq_len, 14]
        change_types = torch.cat([change_features.unsqueeze(0)] * 10, dim=0)  # Repeat for sequence length
        
        # Security features: [seq_len, 12]
        security_features_expanded = torch.cat([security_features.unsqueeze(0)] * 10, dim=0)  # Repeat for sequence length
        
        # Complexity features: [seq_len, 10]
        complexity_features_expanded = torch.cat([complexity_features.unsqueeze(0)] * 10, dim=0)  # Repeat for sequence length
        
        return {
            'ast_nodes': ast_nodes,
            'context_info': context_info,
            'patterns': patterns,
            'change_types': change_types,
            'security_features': security_features_expanded,
            'complexity_features': complexity_features_expanded,
            # Other features would be added similarly in a full implementation
        }


def extract_features_for_dataset(diffs: List[str], messages: List[str]) -> List[Dict[str, torch.Tensor]]:
    """
    Extract multimodal features for an entire dataset.
    
    Args:
        diffs: List of code diffs
        messages: List of commit messages (for context if needed)
        
    Returns:
        List of feature dictionaries, one per data point
    """
    extractor = FeatureExtractor()
    features_list = []
    
    for i, diff in enumerate(diffs):
        # For simplicity, treat the entire diff as the "new code"
        # In a real implementation, you'd have both old and new code
        try:
            # This is a simplified approach - in reality, you'd need both old and new code
            # to properly extract change-based features
            features = extractor.extract_all_features("", diff, 
                                                   context=messages[i] if i < len(messages) else None)
            features_list.append(features)
        except Exception as e:
            print(f"Error extracting features for diff {i}: {e}")
            # Add empty features as fallback
            features_list.append({
                'ast_nodes': torch.zeros(10, 22),
                'context_info': torch.zeros(10, 10),
                'patterns': torch.zeros(10, 23),
                'change_types': torch.zeros(10, 14),
                'security_features': torch.zeros(10, 12),
                'complexity_features': torch.zeros(10, 10),
            })
    
    return features_list


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Example code diff
    old_code = """
def calculate_sum(a, b):
    return a + b
"""
    
    new_code = """
def calculate_sum(a, b):
    # Added input validation
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b
"""
    
    features = extractor.extract_all_features(old_code, new_code)
    
    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: shape {value.shape}")