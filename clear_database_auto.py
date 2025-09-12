#!/usr/bin/env python3
"""
Auto Clear Vector Database Script
Automatically deletes all stored documents from the ChromaDB vector database
"""

import os
import sys
import logging
from vector_db import VectorDatabase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_all_documents():
    """Clear all documents from the vector database"""
    try:
        # Initialize vector database
        logger.info("Initializing vector database connection...")
        vector_db = VectorDatabase()
        
        # Get current document count
        initial_count = vector_db.get_document_count()
        logger.info(f"Found {initial_count} documents in database")
        
        if initial_count == 0:
            logger.info("Database is already empty")
            return True
        
        # Get all document IDs
        logger.info("Retrieving all document IDs...")
        all_docs = vector_db.get_all_documents()
        doc_ids = [doc['id'] for doc in all_docs]
        
        # Delete all documents
        logger.info(f"Deleting {len(doc_ids)} documents...")
        deleted_count = 0
        
        for doc_id in doc_ids:
            try:
                if vector_db.delete_document(doc_id):
                    deleted_count += 1
                    if deleted_count % 10 == 0:
                        logger.info(f"Deleted {deleted_count}/{len(doc_ids)} documents...")
            except Exception as e:
                logger.warning(f"Failed to delete document {doc_id}: {e}")
        
        # Verify deletion
        final_count = vector_db.get_document_count()
        logger.info(f"Deletion complete. Documents deleted: {deleted_count}")
        logger.info(f"Remaining documents: {final_count}")
        
        if final_count == 0:
            logger.info("✅ All documents successfully deleted from vector database")
            return True
        else:
            logger.warning(f"⚠️ {final_count} documents remain in database")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error clearing database: {e}")
        return False

if __name__ == "__main__":
    logger.info("Vector RAG Database - Auto Document Cleaner")
    logger.info("Starting automatic database cleanup")
    
    success = clear_all_documents()
    
    if success:
        logger.info("Database cleanup completed successfully")
    else:
        logger.error("Database cleanup completed with errors")
        sys.exit(1)