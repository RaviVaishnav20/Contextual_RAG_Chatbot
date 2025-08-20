-- Initialize database with PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user if not exists (in case of custom setup)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ravi') THEN
        CREATE USER ravi WITH PASSWORD 'password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE vector_db TO ravi;
GRANT ALL ON SCHEMA public TO ravi;