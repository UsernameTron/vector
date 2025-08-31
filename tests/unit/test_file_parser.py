"""Unit tests for File Parser functionality"""

import pytest
import io
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

# Add src to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.utils.file_parser import FileParser, FileParsingError, parse_uploaded_file


class TestFileParser:
    """Test File Parser functionality"""

    def setup_method(self):
        """Setup for each test method"""
        self.parser = FileParser()

    def test_file_parser_initialization(self):
        """Test FileParser initializes correctly"""
        assert self.parser.documents_generated == 0
        assert self.parser.errors_encountered == []
        assert self.parser.MAX_FILE_SIZE == 50 * 1024 * 1024
        assert self.parser.CHUNK_SIZE == 1000

    def test_validate_file_success(self):
        """Test file validation passes for valid files"""
        small_content = b"test content"
        
        # Test Excel file
        self.parser.validate_file(small_content, "test.xlsx")
        
        # Test CSV file
        self.parser.validate_file(small_content, "test.csv")
        
        # No exceptions should be raised

    def test_validate_file_size_limit(self):
        """Test file validation fails for oversized files"""
        large_content = b"x" * (self.parser.MAX_FILE_SIZE + 1)
        
        with pytest.raises(FileParsingError, match="File size .* exceeds maximum"):
            self.parser.validate_file(large_content, "test.xlsx")

    def test_validate_file_unsupported_extension(self):
        """Test file validation fails for unsupported extensions"""
        content = b"test content"
        
        with pytest.raises(FileParsingError, match="Unsupported file extension"):
            self.parser.validate_file(content, "test.pdf")

    def test_get_file_extension(self):
        """Test file extension extraction"""
        assert self.parser._get_file_extension("test.xlsx") == ".xlsx"
        assert self.parser._get_file_extension("data.csv") == ".csv"
        assert self.parser._get_file_extension("file.TSV") == ".tsv"
        assert self.parser._get_file_extension("noextension") == ""

    def test_clean_column_name(self):
        """Test column name cleaning"""
        assert self.parser._clean_column_name("User Name") == "user_name"
        assert self.parser._clean_column_name("Sales-Data!") == "sales_data"
        assert self.parser._clean_column_name("  Spaced  ") == "spaced"
        assert self.parser._clean_column_name("123Numbers") == "123numbers"
        assert self.parser._clean_column_name("") == "unnamed_column"
        assert self.parser._clean_column_name(None) == "unnamed_column"

    def test_clean_cell_value(self):
        """Test cell value cleaning"""
        from datetime import datetime
        
        # Test string values
        assert self.parser._clean_cell_value("  test  ") == "test"
        assert self.parser._clean_cell_value("multi\n\nlines") == "multi lines"
        
        # Test numeric values
        assert self.parser._clean_cell_value(123) == "123"
        assert self.parser._clean_cell_value(45.67) == "45.67"
        
        # Test datetime values
        dt = datetime(2024, 1, 1, 12, 0, 0)
        assert self.parser._clean_cell_value(dt) == "2024-01-01 12:00:00"
        
        # Test None/NaN
        assert self.parser._clean_cell_value(None) == ""
        assert self.parser._clean_cell_value(pd.NA) == ""

    def test_detect_delimiter(self):
        """Test CSV delimiter detection"""
        # Comma-separated
        csv_content = "name,age,city\nJohn,30,NYC\nJane,25,LA"
        assert self.parser._detect_delimiter(csv_content) == ","
        
        # Semicolon-separated
        csv_content = "name;age;city\nJohn;30;NYC\nJane;25;LA"
        assert self.parser._detect_delimiter(csv_content) == ";"
        
        # Tab-separated
        csv_content = "name\tage\tcity\nJohn\t30\tNYC\nJane\t25\tLA"
        assert self.parser._detect_delimiter(csv_content) == "\t"
        
        # Default fallback
        csv_content = "nameagecity\nJohn30NYC"
        assert self.parser._detect_delimiter(csv_content) == ","

    def test_decode_with_fallback(self):
        """Test encoding fallback mechanism"""
        # UTF-8 content
        utf8_content = "Hello World".encode('utf-8')
        result = self.parser._decode_with_fallback(utf8_content)
        assert result == "Hello World"
        
        # Latin-1 content
        latin1_content = "Café".encode('latin-1')
        result = self.parser._decode_with_fallback(latin1_content)
        assert "Caf" in result  # Should decode somehow

    @patch('chardet.detect')
    def test_detect_encoding(self, mock_detect):
        """Test encoding detection"""
        # High confidence detection
        mock_detect.return_value = {'encoding': 'utf-8', 'confidence': 0.95}
        result = self.parser._detect_encoding(b"test content")
        assert result == 'utf-8'
        
        # Low confidence detection
        mock_detect.return_value = {'encoding': 'latin-1', 'confidence': 0.3}
        result = self.parser._detect_encoding(b"test content")
        assert result == 'utf-8'  # Falls back to UTF-8
        
        # Detection failure
        mock_detect.side_effect = Exception("Detection failed")
        result = self.parser._detect_encoding(b"test content")
        assert result == 'utf-8'  # Falls back to UTF-8

    def test_chunk_dataframe(self):
        """Test DataFrame chunking for large files"""
        # Create test DataFrame
        df = pd.DataFrame({
            'col1': range(100),
            'col2': ['value'] * 100
        })
        
        # Test chunking
        chunks = self.parser._chunk_dataframe(df, chunk_size=25)
        assert len(chunks) == 4
        assert len(chunks[0]) == 25
        assert len(chunks[3]) == 25

    def test_dataframe_to_documents(self):
        """Test DataFrame to documents conversion"""
        # Create test DataFrame
        df = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith'],
            'Age': [30, 25],
            'City': ['New York', 'Los Angeles']
        })
        
        docs = self.parser._dataframe_to_documents(df, 'test.csv', 'csv_upload')
        
        assert len(docs) == 2
        assert 'name: John Doe' in docs[0]['content']
        assert 'age: 30' in docs[0]['content']
        assert 'city: New York' in docs[0]['content']
        assert docs[0]['metadata']['filename'] == 'test.csv'
        assert docs[0]['metadata']['source'] == 'csv_upload'
        assert docs[0]['metadata']['file_type'] == 'csv'

    def test_parse_csv_success(self):
        """Test successful CSV parsing"""
        csv_content = "Name,Age,City\nJohn,30,NYC\nJane,25,LA"
        csv_bytes = csv_content.encode('utf-8')
        
        documents = self.parser.parse_csv(csv_bytes, 'test.csv', 'csv_upload')
        
        assert len(documents) == 2
        assert 'name: John' in documents[0]['content']
        assert documents[0]['metadata']['filename'] == 'test.csv'

    def test_parse_csv_empty_file(self):
        """Test CSV parsing with empty file"""
        csv_content = ""
        csv_bytes = csv_content.encode('utf-8')
        
        with pytest.raises(FileParsingError, match="CSV parsing failed"):
            self.parser.parse_csv(csv_bytes, 'empty.csv', 'csv_upload')

    def test_parse_csv_malformed(self):
        """Test CSV parsing with malformed content"""
        csv_content = "Invalid\nCSV\nContent\nWith\nInconsistent\nColumns"
        csv_bytes = csv_content.encode('utf-8')
        
        # Should not raise an exception but may produce documents
        try:
            documents = self.parser.parse_csv(csv_bytes, 'malformed.csv', 'csv_upload')
            # If parsing succeeds, that's also acceptable
            assert isinstance(documents, list)
        except FileParsingError:
            # If parsing fails, that's expected for malformed data
            pass

    @patch('pandas.ExcelFile')
    @patch('pandas.read_excel')
    def test_parse_excel_success(self, mock_read_excel, mock_excel_file):
        """Test successful Excel parsing"""
        # Mock Excel file with one sheet
        mock_excel_instance = Mock()
        mock_excel_instance.sheet_names = ['Sheet1']
        mock_excel_file.return_value = mock_excel_instance
        
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'Name': ['John', 'Jane'],
            'Age': [30, 25]
        })
        mock_read_excel.return_value = mock_df
        
        excel_content = b"fake excel content"
        documents = self.parser.parse_excel(excel_content, 'test.xlsx', 'excel_upload')
        
        assert len(documents) == 2
        assert documents[0]['metadata']['sheet_name'] == 'Sheet1'

    @patch('pandas.ExcelFile')
    def test_parse_excel_multiple_sheets(self, mock_excel_file):
        """Test Excel parsing with multiple sheets"""
        # Mock Excel file with multiple sheets
        mock_excel_instance = Mock()
        mock_excel_instance.sheet_names = ['Sheet1', 'Sheet2']
        mock_excel_file.return_value = mock_excel_instance
        
        # Mock read_excel to return different DataFrames for different sheets
        def mock_read_excel_side_effect(*args, **kwargs):
            sheet_name = kwargs.get('sheet_name', 'Sheet1')
            if sheet_name == 'Sheet1':
                return pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            elif sheet_name == 'Sheet2':
                return pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
            else:
                return pd.DataFrame()
        
        with patch('pandas.read_excel', side_effect=mock_read_excel_side_effect):
            excel_content = b"fake excel content"
            documents = self.parser.parse_excel(excel_content, 'test.xlsx', 'excel_upload')
            
            # Should have documents from both sheets
            assert len(documents) == 4
            sheet_names = [doc['metadata']['sheet_name'] for doc in documents]
            assert 'Sheet1' in sheet_names
            assert 'Sheet2' in sheet_names

    @patch('pandas.ExcelFile')
    def test_parse_excel_empty_sheet(self, mock_excel_file):
        """Test Excel parsing with empty sheet"""
        mock_excel_instance = Mock()
        mock_excel_instance.sheet_names = ['EmptySheet']
        mock_excel_file.return_value = mock_excel_instance
        
        with patch('pandas.read_excel', return_value=pd.DataFrame()):
            excel_content = b"fake excel content"
            documents = self.parser.parse_excel(excel_content, 'test.xlsx', 'excel_upload')
            
            assert len(documents) == 0

    def test_parse_file_excel(self):
        """Test parse_file method with Excel file"""
        with patch.object(self.parser, 'parse_excel') as mock_parse_excel:
            mock_parse_excel.return_value = [{'content': 'test', 'metadata': {}}]
            
            content = b"fake excel content"
            documents = self.parser.parse_file(content, 'test.xlsx', 'upload')
            
            assert len(documents) == 1
            mock_parse_excel.assert_called_once()

    def test_parse_file_csv(self):
        """Test parse_file method with CSV file"""
        with patch.object(self.parser, 'parse_csv') as mock_parse_csv:
            mock_parse_csv.return_value = [{'content': 'test', 'metadata': {}}]
            
            content = b"fake csv content"
            documents = self.parser.parse_file(content, 'test.csv', 'upload')
            
            assert len(documents) == 1
            mock_parse_csv.assert_called_once()

    def test_get_parsing_stats(self):
        """Test parsing statistics"""
        self.parser.documents_generated = 5
        self.parser.errors_encountered = ['error1', 'error2']
        
        stats = self.parser.get_parsing_stats()
        
        assert stats['documents_generated'] == 5
        assert stats['errors_encountered'] == 2
        assert stats['error_details'] == ['error1', 'error2']

    def test_reset_stats(self):
        """Test parsing statistics reset"""
        self.parser.documents_generated = 5
        self.parser.errors_encountered = ['error1']
        
        self.parser.reset_stats()
        
        assert self.parser.documents_generated == 0
        assert self.parser.errors_encountered == []


class TestParseUploadedFile:
    """Test the convenience function parse_uploaded_file"""
    
    def test_parse_uploaded_file_success(self):
        """Test successful file parsing via convenience function"""
        csv_content = "Name,Age\nJohn,30\nJane,25"
        csv_bytes = csv_content.encode('utf-8')
        
        documents, stats = parse_uploaded_file(csv_bytes, 'test.csv', 'upload')
        
        assert len(documents) == 2
        assert stats['documents_generated'] == 2
        assert stats['errors_encountered'] == 0

    def test_parse_uploaded_file_error(self):
        """Test file parsing error handling"""
        invalid_content = b"invalid content"
        
        with pytest.raises(FileParsingError):
            parse_uploaded_file(invalid_content, 'test.pdf', 'upload')


class TestFileParsingIntegration:
    """Integration tests for file parsing with real data"""
    
    def test_real_csv_parsing(self):
        """Test parsing a real CSV structure"""
        csv_data = """ID,Name,Email,Department,Salary
1,John Doe,john@company.com,Engineering,75000
2,Jane Smith,jane@company.com,Marketing,65000
3,Bob Johnson,bob@company.com,Sales,55000"""
        
        parser = FileParser()
        csv_bytes = csv_data.encode('utf-8')
        documents = parser.parse_csv(csv_bytes, 'employees.csv', 'hr_upload')
        
        assert len(documents) == 3
        
        # Check first document
        first_doc = documents[0]
        assert 'id: 1' in first_doc['content']
        assert 'name: John Doe' in first_doc['content']
        assert 'email: john@company.com' in first_doc['content']
        assert first_doc['metadata']['filename'] == 'employees.csv'
        assert first_doc['metadata']['row_index'] == 0  # pandas index starts at 0

    def test_csv_with_special_characters(self):
        """Test CSV parsing with special characters and encoding"""
        csv_data = """Name,Description,Price
"Café Latte","Hot coffee with milk",€3.50
"Crème Brûlée","French dessert",€6.00"""
        
        parser = FileParser()
        csv_bytes = csv_data.encode('utf-8')
        documents = parser.parse_csv(csv_bytes, 'menu.csv', 'restaurant')
        
        assert len(documents) == 2
        assert 'Café Latte' in documents[0]['content']
        assert '€3.50' in documents[0]['content']

    def test_large_dataset_chunking(self):
        """Test chunking with a larger dataset"""
        # Create a large CSV with more than CHUNK_SIZE rows
        rows = []
        rows.append("ID,Name,Value")
        for i in range(1500):  # More than default chunk size of 1000
            rows.append(f"{i},Name{i},Value{i}")
        
        csv_data = "\n".join(rows)
        
        parser = FileParser()
        csv_bytes = csv_data.encode('utf-8')
        documents = parser.parse_csv(csv_bytes, 'large_data.csv', 'bulk_upload')
        
        assert len(documents) == 1500
        
        # Check that chunking metadata is present
        chunk_0_docs = [doc for doc in documents if doc['metadata']['chunk'] == 0]
        chunk_1_docs = [doc for doc in documents if doc['metadata']['chunk'] == 1]
        
        assert len(chunk_0_docs) == 1000
        assert len(chunk_1_docs) == 500

    def test_excel_like_csv_structure(self):
        """Test CSV that mimics Excel structure with merged-like content"""
        csv_data = """Quarter,Revenue,Expenses,Profit
Q1 2024,150000,100000,50000
Q2 2024,180000,120000,60000
,,,
Summary,330000,220000,110000"""
        
        parser = FileParser()
        csv_bytes = csv_data.encode('utf-8')
        documents = parser.parse_csv(csv_bytes, 'financial.csv', 'finance')
        
        # Should handle empty rows gracefully
        assert len(documents) >= 2  # At minimum Q1, Q2, and Summary rows with content