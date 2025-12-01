"""
RAG (Retrieval-Augmented Generation) System using ChromaDB and Sentence Transformers.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}. "
        "Install with: pip install chromadb sentence-transformers"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG system for document retrieval and question answering.
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the RAG system.

        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Sentence transformer model name
            persist_directory: Directory to persist ChromaDB data
        """
        try:
            logger.info(f"Initializing RAG system with model: {model_name}")

            # Initialize sentence transformer model
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")

            # Initialize ChromaDB client
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Document embeddings for RAG"}
            )
            logger.info(f"ChromaDB collection '{collection_name}' ready")

        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def load_documents(self, directory: str) -> List[Dict[str, str]]:
        """
        Load and chunk all .txt files from a directory.

        Args:
            directory: Path to directory containing text documents

        Returns:
            List of document chunks with metadata

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no valid documents found
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            if not dir_path.is_dir():
                raise ValueError(f"Path is not a directory: {directory}")

            all_chunks = []
            supported_extensions = ['.txt']

            # Find all supported files
            files = [
                f for f in dir_path.rglob('*')
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]

            if not files:
                raise ValueError(
                    f"No .txt files found in {directory}"
                )

            logger.info(f"Found {len(files)} documents to process")

            # Process each file
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.strip():
                        logger.warning(f"Skipping empty file: {file_path}")
                        continue

                    # Chunk the document
                    chunks = self._chunk_text(content)

                    # Add metadata to each chunk
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            'text': chunk,
                            'source': str(file_path),
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        })

                    logger.info(
                        f"Processed {file_path.name}: {len(chunks)} chunks"
                    )

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

            if not all_chunks:
                raise ValueError("No valid content extracted from documents")

            logger.info(
                f"Total chunks created: {len(all_chunks)} from {len(files)} files"
            )
            return all_chunks

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks while preserving sentence boundaries.

        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in tokens (approximate)
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of text chunks
        """
        try:
            # Split into sentences using regex
            sentences = re.split(r'(?<=[.!?])\s+', text)

            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Rough token count (words as proxy)
                sentence_length = len(sentence.split())

                # If single sentence exceeds chunk_size, split it
                if sentence_length > chunk_size:
                    # Save current chunk if it exists
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    # Split long sentence into smaller parts
                    words = sentence.split()
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = ' '.join(words[i:i + chunk_size])
                        chunks.append(chunk)
                    continue

                # Check if adding this sentence exceeds chunk_size
                if current_length + sentence_length > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))

                    # Overlap: keep last few sentences
                    overlap_text = ' '.join(current_chunk[-2:])
                    overlap_length = len(overlap_text.split())

                    if overlap_length < overlap:
                        current_chunk = current_chunk[-2:]
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0

                current_chunk.append(sentence)
                current_length += sentence_length

            # Add the last chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks if chunks else [text]

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return [text]

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        Generate embeddings for chunks and add to ChromaDB.

        Args:
            chunks: List of document chunks with metadata

        Raises:
            ValueError: If chunks list is empty
        """
        try:
            if not chunks:
                raise ValueError("No chunks provided")

            logger.info(f"Adding {len(chunks)} chunks to ChromaDB")

            # Extract texts and metadata
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [
                {
                    'source': chunk['source'],
                    'chunk_id': str(chunk['chunk_id']),
                    'total_chunks': str(chunk['total_chunks'])
                }
                for chunk in chunks
            ]

            # Generate unique IDs
            ids = [
                f"{chunk['source']}_{chunk['chunk_id']}"
                for chunk in chunks
            ]

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # Add to ChromaDB in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))

                self.collection.add(
                    embeddings=embeddings[i:end_idx].tolist(),
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )

                logger.info(
                    f"Added batch {i//batch_size + 1}: "
                    f"{end_idx}/{len(chunks)} chunks"
                )

            logger.info(
                f"Successfully added {len(chunks)} chunks to collection"
            )

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def query(
        self,
        question: str,
        top_k: int = 3
    ) -> List[Tuple[str, Dict[str, str], float]]:
        """
        Query the RAG system with a question.

        Args:
            question: Question to search for
            top_k: Number of top results to return

        Returns:
            List of tuples: (text, metadata, distance/score)

        Raises:
            ValueError: If question is empty or top_k is invalid
        """
        try:
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")

            if top_k < 1:
                raise ValueError("top_k must be at least 1")

            # Check if collection has documents
            collection_count = self.collection.count()
            if collection_count == 0:
                raise ValueError(
                    "Collection is empty. Please load documents first using "
                    "load_documents() and add_documents() methods."
                )

            logger.info(f"Querying: '{question}' (top_k={top_k})")

            # Generate query embedding
            query_embedding = self.model.encode(
                question,
                convert_to_numpy=True
            )

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, collection_count)
            )

            # Format results
            formatted_results = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    text = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 0.0

                    formatted_results.append((text, metadata, distance))

            logger.info(f"Found {len(formatted_results)} relevant chunks")
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            raise

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"description": "Document embeddings for RAG"}
            )
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_chunks': 0, 'collection_name': 'unknown'}


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize RAG system
        rag = RAGSystem()

        # Load documents from a directory
        # chunks = rag.load_documents("./data")
        # rag.add_documents(chunks)

        # Query example
        # results = rag.query("What is PHI?", top_k=3)
        # for text, metadata, score in results:
        #     print(f"Score: {score}")
        #     print(f"Source: {metadata['source']}")
        #     print(f"Text: {text[:200]}...")
        #     print("-" * 80)

        print("RAG System initialized successfully")
        print(rag.get_collection_stats())

    except Exception as e:
        logger.error(f"Error in main: {e}")
