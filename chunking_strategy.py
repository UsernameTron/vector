"""
Document Chunking Strategy
Intelligent document chunking for optimal vector search and RAG performance
"""

from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Different chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    HYBRID = "hybrid"

@dataclass
class ChunkConfig:
    """Configuration for document chunking"""
    chunk_size: int = 512  # Target chunk size in tokens
    overlap: int = 50      # Overlap between chunks in tokens
    min_chunk_size: int = 50  # Minimum viable chunk size
    max_chunk_size: int = 1024  # Maximum chunk size
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True

@dataclass
class DocumentChunk:
    """Represents a document chunk"""
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    token_count: int
    char_start: int
    char_end: int
    overlap_prev: int = 0
    overlap_next: int = 0

class DocumentChunker:
    """Advanced document chunking with multiple strategies"""
    
    def __init__(self, config: ChunkConfig = None):
        """
        Initialize document chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tiktoken, using fallback: {e}")
            self.tokenizer = None
        
        # Compile regex patterns for boundary detection
        self.sentence_endings = re.compile(r'[.!?]+[\s\n]+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        self.section_breaks = re.compile(r'\n\s*#{1,6}\s+[^\n]+\n')  # Markdown headers
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return len(text) // 4
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to tokens"""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # Fallback: character-based chunking
            return list(range(len(text)))
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text"""
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        else:
            # Fallback: return as string representation
            return ''.join([chr(min(t, 127)) for t in tokens])
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """Find sentence boundary positions in text"""
        boundaries = [0]  # Start of text
        
        for match in self.sentence_endings.finditer(text):
            end_pos = match.end()
            if end_pos < len(text):
                boundaries.append(end_pos)
        
        if boundaries[-1] != len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def find_paragraph_boundaries(self, text: str) -> List[int]:
        """Find paragraph boundary positions in text"""
        boundaries = [0]  # Start of text
        
        for match in self.paragraph_breaks.finditer(text):
            end_pos = match.end()
            if end_pos < len(text):
                boundaries.append(end_pos)
        
        if boundaries[-1] != len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def find_section_boundaries(self, text: str) -> List[int]:
        """Find section boundary positions (Markdown headers, etc.)"""
        boundaries = [0]  # Start of text
        
        for match in self.section_breaks.finditer(text):
            start_pos = match.start()
            if start_pos > 0:
                boundaries.append(start_pos)
        
        if boundaries[-1] != len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def chunk_fixed_size(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document using fixed token size"""
        tokens = self.encode_text(text)
        chunks = []
        
        for i in range(0, len(tokens), self.config.chunk_size - self.config.overlap):
            start_idx = max(0, i - self.config.overlap) if i > 0 else 0
            end_idx = min(len(tokens), i + self.config.chunk_size)
            
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.decode_tokens(chunk_tokens)
            
            # Skip chunks that are too small
            if len(chunk_tokens) < self.config.min_chunk_size:
                continue
            
            # Calculate character positions (approximate)
            char_start = int(start_idx * len(text) / len(tokens))
            char_end = int(end_idx * len(text) / len(tokens))
            
            chunk = DocumentChunk(
                content=chunk_text.strip(),
                metadata={
                    **metadata,
                    'chunk_index': len(chunks),
                    'chunk_type': 'fixed_size',
                    'token_start': start_idx,
                    'token_end': end_idx,
                    'original_doc_id': metadata.get('doc_id'),
                    'total_chunks': None  # Will be set after all chunks are created
                },
                chunk_index=len(chunks),
                token_count=len(chunk_tokens),
                char_start=char_start,
                char_end=char_end,
                overlap_prev=self.config.overlap if i > 0 else 0,
                overlap_next=self.config.overlap if end_idx < len(tokens) else 0
            )
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def chunk_sentence_boundary(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document respecting sentence boundaries"""
        sentence_boundaries = self.find_sentence_boundaries(text)
        chunks = []
        
        current_chunk = ""
        current_start = 0
        
        for i in range(len(sentence_boundaries) - 1):
            sentence = text[sentence_boundaries[i]:sentence_boundaries[i + 1]]
            potential_chunk = current_chunk + sentence
            
            token_count = self.count_tokens(potential_chunk)
            
            if token_count <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is ready, start new one
                if current_chunk.strip() and self.count_tokens(current_chunk) >= self.config.min_chunk_size:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={
                            **metadata,
                            'chunk_index': len(chunks),
                            'chunk_type': 'sentence_boundary',
                            'original_doc_id': metadata.get('doc_id')
                        },
                        chunk_index=len(chunks),
                        token_count=self.count_tokens(current_chunk),
                        char_start=current_start,
                        char_end=sentence_boundaries[i]
                    )
                    chunks.append(chunk)
                
                current_chunk = sentence
                current_start = sentence_boundaries[i]
        
        # Add final chunk
        if current_chunk.strip() and self.count_tokens(current_chunk) >= self.config.min_chunk_size:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata={
                    **metadata,
                    'chunk_index': len(chunks),
                    'chunk_type': 'sentence_boundary',
                    'original_doc_id': metadata.get('doc_id')
                },
                chunk_index=len(chunks),
                token_count=self.count_tokens(current_chunk),
                char_start=current_start,
                char_end=len(text)
            )
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def chunk_hybrid(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Hybrid chunking strategy that combines multiple approaches
        1. Try to respect paragraph boundaries
        2. Fall back to sentence boundaries
        3. Final fallback to fixed-size with smart splitting
        """
        # First, try paragraph-based chunking
        paragraph_boundaries = self.find_paragraph_boundaries(text)
        
        if len(paragraph_boundaries) > 2:  # More than just start and end
            chunks = self._chunk_by_boundaries(text, paragraph_boundaries, metadata, 'paragraph')
            if chunks:
                return chunks
        
        # Fall back to sentence-based chunking
        try:
            return self.chunk_sentence_boundary(text, metadata)
        except Exception as e:
            logger.warning(f"Sentence boundary chunking failed: {e}")
        
        # Final fallback to fixed-size
        return self.chunk_fixed_size(text, metadata)
    
    def _chunk_by_boundaries(self, text: str, boundaries: List[int], 
                           metadata: Dict[str, Any], chunk_type: str) -> List[DocumentChunk]:
        """Generic method to chunk text by given boundaries"""
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for i in range(len(boundaries) - 1):
            segment = text[boundaries[i]:boundaries[i + 1]]
            potential_chunk = current_chunk + segment
            
            token_count = self.count_tokens(potential_chunk)
            
            if token_count <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is ready
                if current_chunk.strip() and self.count_tokens(current_chunk) >= self.config.min_chunk_size:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={
                            **metadata,
                            'chunk_index': len(chunks),
                            'chunk_type': chunk_type,
                            'original_doc_id': metadata.get('doc_id')
                        },
                        chunk_index=len(chunks),
                        token_count=self.count_tokens(current_chunk),
                        char_start=current_start,
                        char_end=boundaries[i]
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = segment
                current_start = boundaries[i]
        
        # Add final chunk
        if current_chunk.strip() and self.count_tokens(current_chunk) >= self.config.min_chunk_size:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata={
                    **metadata,
                    'chunk_index': len(chunks),
                    'chunk_type': chunk_type,
                    'original_doc_id': metadata.get('doc_id')
                },
                chunk_index=len(chunks),
                token_count=self.count_tokens(current_chunk),
                char_start=current_start,
                char_end=len(text)
            )
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Main method to chunk a document based on configured strategy
        
        Args:
            text: Document text to chunk
            metadata: Document metadata
            
        Returns:
            List of document chunks
        """
        if not text or not text.strip():
            return []
        
        if metadata is None:
            metadata = {}
        
        # Add chunking metadata
        metadata['chunking_config'] = {
            'strategy': self.config.strategy.value,
            'chunk_size': self.config.chunk_size,
            'overlap': self.config.overlap,
            'min_chunk_size': self.config.min_chunk_size,
            'max_chunk_size': self.config.max_chunk_size
        }
        
        try:
            if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
                return self.chunk_fixed_size(text, metadata)
            elif self.config.strategy == ChunkingStrategy.SENTENCE_BOUNDARY:
                return self.chunk_sentence_boundary(text, metadata)
            elif self.config.strategy == ChunkingStrategy.HYBRID:
                return self.chunk_hybrid(text, metadata)
            else:
                # Default to hybrid
                return self.chunk_hybrid(text, metadata)
                
        except Exception as e:
            logger.error(f"Chunking failed with strategy {self.config.strategy}: {e}")
            # Fallback to fixed size
            try:
                return self.chunk_fixed_size(text, metadata)
            except Exception as fallback_error:
                logger.error(f"Fallback chunking also failed: {fallback_error}")
                return []
    
    def get_chunk_context(self, chunks: List[DocumentChunk], target_chunk_index: int, 
                         context_size: int = 1) -> str:
        """
        Get context around a specific chunk by including neighboring chunks
        
        Args:
            chunks: List of all chunks
            target_chunk_index: Index of the target chunk
            context_size: Number of chunks to include before and after
            
        Returns:
            Combined text with context
        """
        if not chunks or target_chunk_index < 0 or target_chunk_index >= len(chunks):
            return ""
        
        start_idx = max(0, target_chunk_index - context_size)
        end_idx = min(len(chunks), target_chunk_index + context_size + 1)
        
        context_chunks = chunks[start_idx:end_idx]
        context_text = "\n\n".join([chunk.content for chunk in context_chunks])
        
        return context_text
    
    def analyze_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Analyze chunking results for optimization"""
        if not chunks:
            return {'total_chunks': 0}
        
        token_counts = [chunk.token_count for chunk in chunks]
        content_lengths = [len(chunk.content) for chunk in chunks]
        
        analysis = {
            'total_chunks': len(chunks),
            'token_stats': {
                'min': min(token_counts),
                'max': max(token_counts),
                'mean': sum(token_counts) / len(token_counts),
                'total': sum(token_counts)
            },
            'content_stats': {
                'min_length': min(content_lengths),
                'max_length': max(content_lengths),
                'mean_length': sum(content_lengths) / len(content_lengths),
                'total_length': sum(content_lengths)
            },
            'chunk_types': {}
        }
        
        # Analyze chunk types
        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            analysis['chunk_types'][chunk_type] = analysis['chunk_types'].get(chunk_type, 0) + 1
        
        # Calculate efficiency metrics
        total_original_tokens = sum(token_counts)
        if total_original_tokens > 0:
            overlap_tokens = sum([chunk.overlap_prev + chunk.overlap_next for chunk in chunks])
            analysis['efficiency'] = {
                'overlap_ratio': overlap_tokens / total_original_tokens,
                'compression_ratio': len(chunks) / (total_original_tokens / self.config.chunk_size)
            }
        
        return analysis

# Convenience function for easy chunking
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50, 
               strategy: ChunkingStrategy = ChunkingStrategy.HYBRID) -> List[str]:
    """
    Simple function to chunk text and return content strings
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens
        strategy: Chunking strategy
        
    Returns:
        List of chunk contents
    """
    config = ChunkConfig(chunk_size=chunk_size, overlap=overlap, strategy=strategy)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(text)
    return [chunk.content for chunk in chunks]