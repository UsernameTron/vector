"""
File Parser Module for Excel and CSV Processing
Handles Excel (.xlsx, .xls) and CSV/TSV files for Vector RAG Database
"""

import pandas as pd
import io
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import chardet
import re
import uuid

logger = logging.getLogger(__name__)


class FileParsingError(Exception):
    """Custom exception for file parsing errors"""
    pass


class FileParser:
    """Handles parsing of Excel and CSV files into structured documents"""
    
    # Maximum file size (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # Chunk size for large files
    CHUNK_SIZE = 1000
    
    # Supported file extensions
    EXCEL_EXTENSIONS = {'.xlsx', '.xls', '.xlsm'}
    CSV_EXTENSIONS = {'.csv', '.tsv', '.txt'}
    
    # Common CSV delimiters to try
    CSV_DELIMITERS = [',', ';', '\t', '|']
    
    # Encoding options to try
    ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
    
    def __init__(self):
        """Initialize the file parser"""
        self.documents_generated = 0
        self.errors_encountered = []

    def validate_file(self, file_content: bytes, filename: str) -> None:
        """
        Validate file size and extension
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Raises:
            FileParsingError: If file validation fails
        """
        # Check file size
        if len(file_content) > self.MAX_FILE_SIZE:
            raise FileParsingError(f"File size ({len(file_content)} bytes) exceeds maximum allowed size ({self.MAX_FILE_SIZE} bytes)")
        
        # Check file extension
        file_ext = self._get_file_extension(filename)
        if file_ext not in (self.EXCEL_EXTENSIONS | self.CSV_EXTENSIONS):
            raise FileParsingError(f"Unsupported file extension: {file_ext}")
        
        logger.info(f"File validation passed: {filename} ({len(file_content)} bytes)")

    def parse_file(self, file_content: bytes, filename: str, source: str = "file_upload") -> List[Dict[str, Any]]:
        """
        Parse file based on its extension
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            source: Source identifier for metadata
            
        Returns:
            List of document dictionaries ready for vector database
        """
        self.validate_file(file_content, filename)
        
        file_ext = self._get_file_extension(filename)
        
        try:
            if file_ext in self.EXCEL_EXTENSIONS:
                return self.parse_excel(file_content, filename, source)
            elif file_ext in self.CSV_EXTENSIONS:
                return self.parse_csv(file_content, filename, source)
            else:
                raise FileParsingError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error parsing file {filename}: {str(e)}")
            raise FileParsingError(f"Failed to parse {filename}: {str(e)}")

    def parse_excel(self, file_content: bytes, filename: str, source: str = "excel_upload") -> List[Dict[str, Any]]:
        """
        Parse Excel file (.xlsx, .xls, .xlsm)
        
        Args:
            file_content: Raw Excel file bytes
            filename: Original filename
            source: Source identifier
            
        Returns:
            List of documents from all sheets
        """
        documents = []
        file_buffer = io.BytesIO(file_content)
        
        try:
            # Read Excel file with all sheets
            excel_file = pd.ExcelFile(file_buffer)
            logger.info(f"Excel file {filename} contains {len(excel_file.sheet_names)} sheets")
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read each sheet
                    df = pd.read_excel(file_buffer, sheet_name=sheet_name, na_values=['', 'N/A', 'NULL', 'null'])
                    
                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' in {filename} is empty")
                        continue
                    
                    # Convert sheet to documents
                    sheet_docs = self._dataframe_to_documents(
                        df, 
                        filename=filename, 
                        source=source,
                        sheet_name=sheet_name
                    )
                    
                    documents.extend(sheet_docs)
                    logger.info(f"Processed sheet '{sheet_name}': {len(sheet_docs)} documents")
                    
                except Exception as e:
                    logger.error(f"Error processing sheet '{sheet_name}' in {filename}: {str(e)}")
                    self.errors_encountered.append(f"Sheet '{sheet_name}': {str(e)}")
                    continue
            
            logger.info(f"Successfully parsed Excel file {filename}: {len(documents)} total documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to parse Excel file {filename}: {str(e)}")
            raise FileParsingError(f"Excel parsing failed: {str(e)}")

    def parse_csv(self, file_content: bytes, filename: str, source: str = "csv_upload") -> List[Dict[str, Any]]:
        """
        Parse CSV/TSV file with automatic delimiter and encoding detection
        
        Args:
            file_content: Raw CSV file bytes
            filename: Original filename
            source: Source identifier
            
        Returns:
            List of documents from CSV
        """
        # Detect encoding
        encoding = self._detect_encoding(file_content)
        logger.info(f"Detected encoding for {filename}: {encoding}")
        
        # Convert to string
        try:
            text_content = file_content.decode(encoding)
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode with {encoding}, trying fallback encodings")
            text_content = self._decode_with_fallback(file_content)
        
        # Detect delimiter
        delimiter = self._detect_delimiter(text_content)
        logger.info(f"Detected delimiter for {filename}: '{delimiter}'")
        
        try:
            # Read CSV with detected parameters
            df = pd.read_csv(
                io.StringIO(text_content),
                delimiter=delimiter,
                na_values=['', 'N/A', 'NULL', 'null', 'nan'],
                encoding=encoding,
                skipinitialspace=True,
                quotechar='"',
                escapechar='\\'
            )
            
            if df.empty:
                raise FileParsingError(f"CSV file {filename} is empty or contains no data")
            
            # Convert to documents
            documents = self._dataframe_to_documents(
                df, 
                filename=filename, 
                source=source
            )
            
            logger.info(f"Successfully parsed CSV file {filename}: {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to parse CSV file {filename}: {str(e)}")
            raise FileParsingError(f"CSV parsing failed: {str(e)}")

    def _dataframe_to_documents(self, df: pd.DataFrame, filename: str, source: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to document format for vector database
        
        Args:
            df: Pandas DataFrame
            filename: Original filename
            source: Source identifier
            sheet_name: Excel sheet name (if applicable)
            
        Returns:
            List of document dictionaries
        """
        documents = []
        timestamp = datetime.now().isoformat()
        
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Process in chunks if file is large
        total_rows = len(df)
        chunks = self._chunk_dataframe(df, self.CHUNK_SIZE)
        
        for chunk_idx, chunk_df in enumerate(chunks):
            chunk_start = chunk_idx * self.CHUNK_SIZE
            chunk_end = min(chunk_start + len(chunk_df), total_rows)
            
            for idx, row in chunk_df.iterrows():
                try:
                    # Create document content
                    content_parts = []
                    metadata = {
                        'filename': filename,
                        'source': source,
                        'row_index': int(idx),
                        'chunk': chunk_idx,
                        'timestamp': timestamp,
                        'file_type': 'excel' if sheet_name else 'csv',
                        'total_rows': total_rows
                    }
                    
                    if sheet_name:
                        metadata['sheet_name'] = sheet_name
                    
                    # Process each column
                    for col_name in df.columns:
                        cell_value = row[col_name]
                        
                        # Skip empty/null values
                        if pd.isna(cell_value) or cell_value == '':
                            continue
                        
                        # Convert value to string and clean it
                        str_value = self._clean_cell_value(cell_value)
                        
                        if str_value:
                            content_parts.append(f"{col_name}: {str_value}")
                            # Add column data to metadata
                            metadata[f'col_{col_name.lower()}'] = str_value[:100]  # Truncate for metadata
                    
                    # Create document content
                    if content_parts:
                        content = " | ".join(content_parts)
                        
                        document = {
                            'id': str(uuid.uuid4()),
                            'content': content,
                            'metadata': metadata,
                            'title': f"{filename} - Row {idx + 1}" + (f" ({sheet_name})" if sheet_name else ""),
                            'source': source
                        }
                        
                        documents.append(document)
                        self.documents_generated += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {str(e)}")
                    self.errors_encountered.append(f"Row {idx}: {str(e)}")
                    continue
        
        return documents

    def _detect_encoding(self, file_content: bytes) -> str:
        """Detect file encoding using chardet"""
        try:
            detected = chardet.detect(file_content[:10000])  # Sample first 10KB
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 0)
            
            if confidence < 0.7:  # Low confidence
                logger.warning(f"Low encoding confidence ({confidence:.2f}) for detected encoding: {encoding}")
                return 'utf-8'  # Fallback to UTF-8
            
            return encoding.lower()
        except Exception:
            logger.warning("Failed to detect encoding, defaulting to utf-8")
            return 'utf-8'

    def _decode_with_fallback(self, file_content: bytes) -> str:
        """Try multiple encodings to decode file content"""
        for encoding in self.ENCODINGS:
            try:
                return file_content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # Last resort: decode with errors='ignore'
        logger.warning("All encoding attempts failed, decoding with errors ignored")
        return file_content.decode('utf-8', errors='ignore')

    def _detect_delimiter(self, text_content: str) -> str:
        """Detect CSV delimiter by analyzing the first few lines"""
        lines = text_content.split('\n')[:5]  # Check first 5 lines
        delimiter_counts = {}
        
        for delimiter in self.CSV_DELIMITERS:
            total_count = 0
            line_counts = []
            
            for line in lines:
                count = line.count(delimiter)
                line_counts.append(count)
                total_count += count
            
            # Check consistency across lines
            if line_counts and all(count == line_counts[0] for count in line_counts) and line_counts[0] > 0:
                delimiter_counts[delimiter] = total_count
        
        if delimiter_counts:
            # Return delimiter with highest count
            best_delimiter = max(delimiter_counts.keys(), key=lambda k: delimiter_counts[k])
            return best_delimiter
        
        # Default to comma
        logger.warning("Could not detect delimiter, defaulting to comma")
        return ','

    def _clean_column_name(self, col_name: str) -> str:
        """Clean and normalize column names"""
        if pd.isna(col_name):
            return 'unnamed_column'
        
        # Convert to string and strip whitespace
        clean_name = str(col_name).strip()
        
        # Replace spaces and some special chars with underscores, but keep hyphens
        clean_name = re.sub(r'[!@#$%^&*()+=\[\]{};\':"\\|,.<>?/~`]', '', clean_name)
        clean_name = re.sub(r'\s+', '_', clean_name)
        clean_name = clean_name.replace('-', '_')
        clean_name = clean_name.lower()
        
        return clean_name if clean_name else 'unnamed_column'

    def _clean_cell_value(self, value: Any) -> str:
        """Clean and convert cell values to searchable text"""
        if pd.isna(value):
            return ''
        
        # Handle different data types
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return ''
            return str(value)
        
        elif isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        
        elif hasattr(value, 'strftime'):  # Other date-like objects
            try:
                return value.strftime('%Y-%m-%d')
            except:
                return str(value)
        
        # Convert to string and clean
        str_value = str(value).strip()
        
        # Remove excessive whitespace
        str_value = re.sub(r'\s+', ' ', str_value)
        
        # Only remove truly non-printable control characters, preserve Unicode
        str_value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str_value)
        
        return str_value

    def _chunk_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Split DataFrame into chunks for processing large files"""
        chunks = []
        total_rows = len(df)
        
        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks

    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        if '.' not in filename:
            return ''
        ext = '.' + filename.lower().split('.')[-1]
        return ext

    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get statistics from the parsing process"""
        return {
            'documents_generated': self.documents_generated,
            'errors_encountered': len(self.errors_encountered),
            'error_details': self.errors_encountered
        }

    def reset_stats(self) -> None:
        """Reset parsing statistics"""
        self.documents_generated = 0
        self.errors_encountered = []


def parse_uploaded_file(file_content: bytes, filename: str, source: str = "file_upload") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function to parse uploaded files
    
    Args:
        file_content: Raw file bytes
        filename: Original filename
        source: Source identifier
        
    Returns:
        Tuple of (documents, parsing_stats)
    """
    parser = FileParser()
    
    try:
        documents = parser.parse_file(file_content, filename, source)
        stats = parser.get_parsing_stats()
        
        return documents, stats
    
    except Exception as e:
        logger.error(f"File parsing failed: {str(e)}")
        raise FileParsingError(f"Failed to parse {filename}: {str(e)}")