import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values, Json
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

class KnowledgeBaseVectorizer:
    def __init__(self, knowledge_base_path: str, db_config: Dict[str, str]):
        """
        Initialize the Knowledge Base Vectorizer
        
        Args:
            knowledge_base_path: Path to the knowledge-base folder
            db_config: Dictionary containing database configuration
                       {host, port, database, user, password}
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.db_config = db_config
        self.client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
        self.conn = None
        self.cursor = None
        
        # Text splitter configuration - increased chunk size for better semantic context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def connect_db(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            self.cursor = self.conn.cursor()
            print("✓ Connected to PostgreSQL database")
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            sys.exit(1)
    
    def setup_pgvector(self):
        """Enable pgvector extension and create table"""
        try:
            # Enable pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table for embeddings
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base_embeddings (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for faster similarity search
            # Note: Index will be created after data insertion for better accuracy
            self.cursor.execute("""
                DROP INDEX IF EXISTS knowledge_base_embeddings_idx;
            """)
            
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS knowledge_base_embeddings_idx 
                ON knowledge_base_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            self.conn.commit()
            print("✓ pgvector extension enabled and table created")
        except Exception as e:
            print(f"✗ Error setting up pgvector: {e}")
            self.conn.rollback()
            sys.exit(1)
    
    def read_markdown_files(self) -> List[Dict]:
        """
        Read all markdown files from knowledge base
        
        Returns:
            List of dictionaries containing file path and content
        """
        documents = []
        
        for file_path in self.knowledge_base_path.rglob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = file_path.relative_to(self.knowledge_base_path)
                    
                    # Extract entity name from filename (e.g., "Long.md" -> "Long")
                    entity_name = file_path.stem
                    
                    # Enrich content with context to reduce ambiguity
                    enriched_content = self._enrich_content(content, relative_path, entity_name)
                    
                    documents.append({
                        'file_path': str(relative_path),
                        'content': enriched_content,
                        'original_content': content,
                        'entity_name': entity_name,
                        'category': relative_path.parts[0] if len(relative_path.parts) > 1 else 'general'
                    })
                    print(f"✓ Read: {relative_path}")
            except Exception as e:
                print(f"✗ Error reading {file_path}: {e}")
        
        return documents
    
    def _enrich_content(self, content: str, relative_path: Path, entity_name: str) -> str:
        """
        Enrich content with contextual information to reduce ambiguity
        
        Args:
            content: Original content
            relative_path: Path relative to knowledge base
            entity_name: Name extracted from filename
            
        Returns:
            Enriched content with better context
        """
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else 'general'
        
        # Add metadata header to provide context
        if category == 'employees':
            prefix = f"[Employee Profile: {entity_name}]\n[Category: Employee Information]\n[Document Type: Personnel Record]\n\n"
        elif category == 'company':
            prefix = f"[Company Information: {entity_name}]\n[Category: Company Information]\n[Document Type: Corporate Documentation]\n\n"
        else:
            prefix = f"[Document: {entity_name}]\n[Category: {category}]\n\n"
        
        return prefix + content
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into chunks
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunked documents with metadata
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            
            for idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    'file_path': doc['file_path'],
                    'chunk_index': idx,
                    'content': chunk,
                    'category': doc['category'],
                    'entity_name': doc.get('entity_name', ''),
                    'total_chunks': len(chunks)
                })
            
            print(f"✓ Chunked {doc['file_path']} into {len(chunks)} chunks")
        
        return chunked_docs
    
    def generate_embeddings(self, texts: List[str], retry_count: int = 3) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API with retry logic
        
        Args:
            texts: List of text strings to embed
            retry_count: Number of retries on failure
            
        Returns:
            List of embedding vectors
        """
        import time
        
        for attempt in range(retry_count):
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",  # Using 1536-d model
                    input=texts
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                print(f"✗ Error generating embeddings (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {retry_count} attempts")
                    return []
    
    def batch_process_embeddings(self, chunks: List[Dict], batch_size: int = 100):
        """
        Process chunks in batches and generate embeddings
        
        Args:
            chunks: List of chunked documents
            batch_size: Number of chunks to process at once
        """
        total_chunks = len(chunks)
        failed_batches = []
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['content'] for chunk in batch]
            
            print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")
            
            embeddings = self.generate_embeddings(texts)
            
            if embeddings and len(embeddings) == len(batch):
                self.save_embeddings(batch, embeddings)
            else:
                print(f"⚠️  Failed to process batch {i//batch_size + 1}")
                failed_batches.append((i, batch))
        
        if failed_batches:
            print(f"\n⚠️  {len(failed_batches)} batches failed. Check your OpenAI API key and rate limits.")
            print("Failed batch indices:", [idx//batch_size + 1 for idx, _ in failed_batches])
    
    def save_embeddings(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Save chunks and embeddings to database
        
        Args:
            chunks: List of chunked documents
            embeddings: Corresponding embedding vectors
        """
        try:
            data = [
                (
                    chunk['file_path'],
                    chunk['chunk_index'],
                    chunk['content'],
                    embedding,
                    Json({
                        'category': chunk['category'],
                        'entity_name': chunk.get('entity_name', ''),
                        'total_chunks': chunk.get('total_chunks', 1)
                    })
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            execute_values(
                self.cursor,
                """
                INSERT INTO knowledge_base_embeddings 
                (file_path, chunk_index, content, embedding, metadata)
                VALUES %s
                """,
                data,
                template="(%s, %s, %s, %s::vector, %s)"
            )
            
            self.conn.commit()
            print(f"✓ Saved {len(chunks)} chunks to database")
        except Exception as e:
            print(f"✗ Error saving embeddings: {e}")
            self.conn.rollback()
    
    def clear_existing_data(self):
        """Clear existing data from the table"""
        try:
            self.cursor.execute("TRUNCATE TABLE knowledge_base_embeddings RESTART IDENTITY;")
            self.conn.commit()
            print("✓ Cleared existing data")
        except Exception as e:
            print(f"✗ Error clearing data: {e}")
            self.conn.rollback()
    
    def check_database_status(self):
        """Check database status and show statistics"""
        try:
            # Check total records
            self.cursor.execute("SELECT COUNT(*) FROM knowledge_base_embeddings;")
            total = self.cursor.fetchone()[0]
            print(f"Total embeddings: {total}")
            
            # Check embeddings with NULL values
            self.cursor.execute("SELECT COUNT(*) FROM knowledge_base_embeddings WHERE embedding IS NULL;")
            null_embeddings = self.cursor.fetchone()[0]
            print(f"NULL embeddings: {null_embeddings}")
            
            if null_embeddings > 0:
                print(f"⚠️  WARNING: {null_embeddings} embeddings are NULL - regenerate with process_knowledge_base(clear_existing=True)")
            
            # Show sample records
            self.cursor.execute("""
                SELECT file_path, chunk_index, LEFT(content, 80), 
                       CASE WHEN embedding IS NULL THEN 'NULL' ELSE 'OK' END as emb_status
                FROM knowledge_base_embeddings 
                LIMIT 3;
            """)
            print("\nSample records:")
            for row in self.cursor.fetchall():
                print(f"  {row[0]} (chunk {row[1]}): {row[2]}... [Embedding: {row[3]}]")
            
            # Check vector dimensions
            self.cursor.execute("""
                SELECT vector_dims(embedding) as dims
                FROM knowledge_base_embeddings 
                WHERE embedding IS NOT NULL
                LIMIT 1;
            """)
            result = self.cursor.fetchone()
            if result:
                print(f"\nEmbedding dimensions: {result[0]}")
            
            # Check if index exists
            self.cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'knowledge_base_embeddings';
            """)
            indexes = self.cursor.fetchall()
            print(f"Indexes: {[idx[0] for idx in indexes]}")
            
        except Exception as e:
            print(f"✗ Error checking database status: {e}")
    
    def process_knowledge_base(self, clear_existing: bool = False):
        """
        Main method to process the entire knowledge base
        
        Args:
            clear_existing: Whether to clear existing data before processing
        """
        print("\n=== Starting Knowledge Base Vectorization ===\n")
        
        # Connect to database
        self.connect_db()
        
        # Setup pgvector
        self.setup_pgvector()
        
        # Clear existing data if requested
        if clear_existing:
            self.clear_existing_data()
        
        # Read markdown files
        print("\n--- Reading markdown files ---")
        documents = self.read_markdown_files()
        print(f"\n✓ Total documents read: {len(documents)}\n")
        
        if not documents:
            print("✗ No documents found to process")
            return
        
        # Chunk documents
        print("--- Chunking documents ---")
        chunks = self.chunk_documents(documents)
        print(f"\n✓ Total chunks created: {len(chunks)}\n")
        
        # Generate embeddings and save
        print("--- Generating embeddings and saving to database ---")
        self.batch_process_embeddings(chunks, batch_size=100)
        
        # Rebuild index for better accuracy after bulk insert
        print("\n--- Rebuilding index for optimal performance ---")
        try:
            self.cursor.execute("REINDEX INDEX knowledge_base_embeddings_idx;")
            self.cursor.execute("ANALYZE knowledge_base_embeddings;")
            self.conn.commit()
            print("✓ Index rebuilt and statistics updated")
        except Exception as e:
            print(f"⚠️  Could not rebuild index: {e}")
        
        print("\n=== Vectorization Complete ===\n")
    
    def search_similar(self, query: str, top_k: int = 5, expand_query: bool = False, 
                      filter_category: str = None, min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for similar content in the knowledge base
        
        Args:
            query: Search query
            top_k: Number of results to return
            expand_query: Whether to expand query for better context matching
            filter_category: Optional category filter (e.g., 'employees', 'company')
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar documents with scores
        """
        # Check if there's data in the database
        self.cursor.execute("SELECT COUNT(*) FROM knowledge_base_embeddings;")
        total_count = self.cursor.fetchone()[0]
        print(f"Total embeddings in database: {total_count}")
        
        if total_count == 0:
            print("⚠️  No embeddings found in database. Please run process_knowledge_base() first.")
            return []
        
        # Intelligent query expansion based on query type
        search_query = query
        if expand_query:
            # Detect query intent and expand accordingly
            if any(word in query.lower() for word in ['who is', 'who are', 'tell me about']):
                search_query = f"Employee profile and information about {query}"
            elif any(word in query.lower() for word in ['backend', 'frontend', 'engineer', 'developer']):
                search_query = f"Employee with job title and skills: {query}"
            elif any(word in query.lower() for word in ['company', 'oven', 'about']):
                search_query = f"Company information and details: {query}"
            else:
                search_query = f"Information about {query}"
            print(f"Expanded query: '{search_query}'")
        
        # Generate query embedding
        print(f"Generating embedding for query: '{search_query}'")
        query_embedding = self.generate_embeddings([search_query])
        
        if not query_embedding:
            print("✗ Failed to generate query embedding")
            return []
        
        query_embedding = query_embedding[0]
        print(f"✓ Query embedding generated (dimension: {len(query_embedding)})")
        
        # Set IVF probes for more accurate search (trades speed for accuracy)
        try:
            self.cursor.execute("SET ivfflat.probes = 10;")
        except Exception as e:
            print(f"⚠️  Could not set ivfflat.probes: {e}")
        
        # Build WHERE clause and parameters
        where_clauses = ["embedding IS NOT NULL"]
        where_params = []
        
        if filter_category:
            where_clauses.append("metadata->>'category' = %s")
            where_params.append(filter_category)
        
        if min_similarity > 0:
            where_clauses.append("(1 - (embedding <=> %s::vector)) >= %s")
            where_params.extend([query_embedding, min_similarity])
        
        where_clause = " AND ".join(where_clauses)
        
        # Build final query parameters
        # Order: SELECT similarity, SELECT distance, WHERE filters, ORDER BY, LIMIT
        params = [query_embedding, query_embedding] + where_params + [query_embedding, top_k]
        
        # Search for similar vectors using cosine distance
        # <=> is cosine distance (0 = identical, 2 = opposite)
        # 1 - cosine_distance = cosine similarity (1 = identical, -1 = opposite)
        query_sql = f"""
            SELECT 
                file_path,
                chunk_index,
                content,
                metadata,
                1 - (embedding <=> %s::vector) as similarity,
                embedding <=> %s::vector as distance
            FROM knowledge_base_embeddings
            WHERE {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        
        self.cursor.execute(query_sql, params)
        
        results = []
        seen_entities = set()  # Track entities to reduce duplicates
        
        for row in self.cursor.fetchall():
            entity_name = row[3].get('entity_name', '') if row[3] else ''
            
            # Group results by entity for better diversity
            result = {
                'file_path': row[0],
                'chunk_index': row[1],
                'content': row[2],
                'metadata': row[3],
                'entity_name': entity_name,
                'similarity': row[4],
                'distance': row[5]
            }
            
            # If we've seen this entity, only include if significantly different
            if entity_name in seen_entities:
                # Check if this chunk is significantly better than previous
                existing = next((r for r in results if r['entity_name'] == entity_name), None)
                if existing and row[4] - existing['similarity'] < 0.05:
                    continue  # Skip very similar chunks from same entity
            
            results.append(result)
            if entity_name:
                seen_entities.add(entity_name)
        
        print(f"✓ Found {len(results)} similar results")
        if results:
            print(f"  Top result similarity: {results[0]['similarity']:.4f} (distance: {results[0]['distance']:.4f})")
            print(f"  Unique entities: {len(seen_entities)}")
        
        return results
    
    def test_exact_match(self, content_snippet: str) -> List[Dict]:
        """
        Test search with exact content from database for debugging
        
        Args:
            content_snippet: A snippet of text that should exist in the database
            
        Returns:
            List of matching results
        """
        print(f"\n🔍 Testing exact match search...")
        print(f"Searching for snippet: '{content_snippet[:100]}...'")
        
        # First check if this content exists in database
        self.cursor.execute("""
            SELECT file_path, chunk_index, LEFT(content, 100)
            FROM knowledge_base_embeddings
            WHERE content ILIKE %s
            LIMIT 1;
        """, (f"%{content_snippet}%",))
        
        match = self.cursor.fetchone()
        if match:
            print(f"✓ Found exact match in database: {match[0]} (chunk {match[1]})")
            print(f"  Content: {match[2]}...")
        else:
            print("✗ No exact text match found in database")
            return []
        
        # Now test semantic search with this content
        results = self.search_similar(content_snippet, top_k=3)
        return results
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Database connection closed")


def main():
    """Main execution function"""
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'ai_chatbot'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    # Path to knowledge base
    knowledge_base_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'knowledge-base'
    )
    print(f"Knowledge Base Path: {knowledge_base_path}")
    
    # Initialize vectorizer
    vectorizer = KnowledgeBaseVectorizer(knowledge_base_path, db_config)
    
    try:
        # Process knowledge base (set clear_existing=True to clear old data)
        vectorizer.process_knowledge_base(clear_existing=True)
        
        # Check database status
        print("\n--- Database Status ---")
        vectorizer.check_database_status()
        
        # Example 1: Search for Backend Engineers
        print("\n--- Test 1: Search for Backend Engineers ---")
        query1 = "Backend Software Engineer"
        results1 = vectorizer.search_similar(
            query1, 
            top_k=5, 
            expand_query=True,
            filter_category='employees'
        )
        
        print(f"\nQuery: {query1}\n")
        if results1:
            for i, result in enumerate(results1, 1):
                entity = result.get('entity_name', 'Unknown')
                print(f"Result {i} (Similarity: {result['similarity']:.4f}) - {entity}:")
                print(f"  File: {result['file_path']}")
                # Remove metadata prefix for cleaner output
                content = result['content'].split('\n\n', 1)[-1] if '\n\n' in result['content'] else result['content']
                print(f"  Content: {content[:200]}...")
                print()
        
        # Example 2: Search for Frontend Engineers
        print("\n--- Test 2: Search for Frontend Engineers ---")
        query2 = "Frontend Software Engineer"
        results2 = vectorizer.search_similar(
            query2, 
            top_k=5, 
            expand_query=True,
            filter_category='employees'
        )
        
        print(f"\nQuery: {query2}\n")
        if results2:
            for i, result in enumerate(results2, 1):
                entity = result.get('entity_name', 'Unknown')
                print(f"Result {i} (Similarity: {result['similarity']:.4f}) - {entity}:")
                print(f"  File: {result['file_path']}")
                content = result['content'].split('\n\n', 1)[-1] if '\n\n' in result['content'] else result['content']
                print(f"  Content: {content[:200]}...")
                print()
        
        # Example 3: Search for specific person
        print("\n--- Test 3: Search for specific person ---")
        query3 = "Who is Long?"
        results3 = vectorizer.search_similar(
            query3, 
            top_k=3, 
            expand_query=True,
            min_similarity=0.3
        )
        
        print(f"\nQuery: {query3}\n")
        if results3:
            for i, result in enumerate(results3, 1):
                entity = result.get('entity_name', 'Unknown')
                print(f"Result {i} (Similarity: {result['similarity']:.4f}) - {entity}:")
                print(f"  File: {result['file_path']}")
                content = result['content'].split('\n\n', 1)[-1] if '\n\n' in result['content'] else result['content']
                print(f"  Content: {content[:200]}...")
                print()
        
        if not results1 and not results2 and not results3:
            print("\n⚠️  No results found. Possible reasons:")
            print("1. Database is empty - run: vectorizer.process_knowledge_base(clear_existing=True)")
            print("2. Embeddings are NULL - check if OpenAI API key is valid")
            print("3. Query doesn't match any content - try different queries")
        
    finally:
        vectorizer.close()


if __name__ == "__main__":
    main()
