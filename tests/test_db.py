#!/usr/bin/env python3
"""
Test database connection and basic operations
"""
from contextual_rag.settings import settings
from sqlalchemy import create_engine, text

def test_connection():
    print(f"Testing connection to: {settings.postgres_dsn}")
    print(f"Table name: {settings.database_table_name}")
    
    try:
        engine = create_engine(settings.postgres_dsn)
        with engine.begin() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version}")
            
            # Test pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("✅ pgvector extension available")
            
            # Test table creation
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {settings.database_table_name} (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT,
                        text TEXT,
                        metadata JSONB,
                        embedding vector(8)
                    )
                    """
                )
            )
            print(f"✅ Table '{settings.database_table_name}' created/verified")
            
            # Test vector operations
            test_vector = "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]"
            conn.execute(
                text(
                    f"""
                    INSERT INTO {settings.database_table_name} 
                    (chunk_id, document_id, text, metadata, embedding)
                    VALUES ('test-1', 'doc-1', 'test text', '{{}}', :embedding)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text
                    """
                ),
                {"embedding": test_vector}
            )
            print("✅ Vector operations working")
            
            # Test query
            result = conn.execute(
                text(
                    f"""
                    SELECT chunk_id, 1 - (embedding <#> :qvec) AS score, text
                    FROM {settings.database_table_name}
                    ORDER BY embedding <#> :qvec
                    LIMIT 1
                    """
                ),
                {"qvec": test_vector}
            ).fetchall()
            print(f"✅ Query test successful: {len(result)} results")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Database connection test passed!")
    else:
        print("\n💥 Database connection test failed!")
