import json
import os
import pandas as pd
from typing import List
from loguru import logger
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from macrec.tools.base import Tool


class RAGTextToSQL(Tool):
    """
    RAG-based Text-to-SQL tool that:
    1. Uses RAG to retrieve relevant database schema/context
    2. Converts natural language to SQL
    3. Executes SQL and returns results
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(config_path, *args, **kwargs)
        
        # Resolve paths relative to project root (where main.py is located)
        # Find project root by looking for common markers (main.py, setup.py, etc.)
        project_root = self._find_project_root(config_path)
        
        # Database connection config
        db_path = self.config.get('db_path', None)
        if db_path and not os.path.isabs(db_path):
            # Resolve relative to project root
            self.db_path = os.path.join(project_root, db_path)
        else:
            self.db_path = db_path
        self.db_type = self.config.get('db_type', 'sqlite')  # sqlite, postgresql, mysql, etc.
        self.connection_string = self.config.get('connection_string', None)
        
        # RAG config
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-3-small')
        self.llm_model = self.config.get('llm_model', 'gpt-4o-mini')
        self.top_k = self.config.get('top_k', 5)
        
        # Schema/documentation path for RAG
        schema_path = self.config.get('schema_path', None)
        if schema_path and not os.path.isabs(schema_path):
            # Resolve relative to project root
            self.schema_path = os.path.join(project_root, schema_path)
        else:
            self.schema_path = schema_path
        self.schema_docs = self.config.get('schema_docs', None)  # Direct schema docs as list
        
        # Log resolved paths for debugging
        logger.debug("RAG SQL Tool initialized:")
        logger.debug(f"  Config path: {config_path}")
        logger.debug(f"  Project root: {project_root}")
        logger.debug(f"  Database path: {self.db_path}")
        logger.debug(f"  Schema path: {self.schema_path}")
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = None
        self._init_vector_store()
        
        # Initialize LLM for text-to-SQL
        self.llm = ChatOpenAI(model=self.llm_model, temperature=0)
        
        # Initialize database connection
        self.db_connection = None
        self._init_db_connection()
        
        # Cache for queries
        self.query_cache = {}
    
    def _find_project_root(self, config_path: str) -> str:
        """
        Find project root directory by looking for common markers.
        Starts from config_path and walks up until finding main.py, setup.py, or .git
        """
        current_dir = os.path.dirname(os.path.abspath(config_path))
        
        # Markers that indicate project root
        markers = ['main.py', 'setup.py', '.git', 'requirements.txt', 'web_demo.py']
        
        while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
            # Check if any marker exists in current directory
            for marker in markers:
                marker_path = os.path.join(current_dir, marker)
                if os.path.exists(marker_path):
                    return current_dir
            # Move up one directory
            current_dir = os.path.dirname(current_dir)
        
        # If no marker found, return directory containing config (fallback)
        return os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    
    def _init_vector_store(self) -> None:
        """Initialize vector store with schema documentation"""
        if self.schema_docs:
            # Use provided schema documents
            documents = [Document(page_content=doc) for doc in self.schema_docs]
        elif self.schema_path:
            # Load schema from file
            import json
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            documents = []
            for table_name, table_info in schema_data.items():
                doc_content = f"Table: {table_name}\n"
                if 'description' in table_info:
                    doc_content += f"Description: {table_info['description']}\n"
                if 'columns' in table_info:
                    doc_content += "Columns:\n"
                    for col_name, col_info in table_info['columns'].items():
                        col_type = col_info.get('type', '')
                        col_desc = col_info.get('description', '')
                        doc_content += f"  - {col_name} ({col_type}): {col_desc}\n"
                if 'examples' in table_info:
                    doc_content += f"Example queries: {table_info['examples']}\n"
                documents.append(Document(page_content=doc_content, metadata={'table': table_name}))
        else:
            # No schema provided, will use empty vector store
            documents = []
        
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Create empty vector store
            self.vector_store = FAISS.from_texts([""], self.embeddings)
    
    def _init_db_connection(self) -> None:
        """Initialize database connection"""
        try:
            if self.db_type == 'sqlite' and self.db_path:
                import sqlite3
                # Enable multi-threading support by setting check_same_thread=False
                # This allows the connection to be used from different threads
                self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            elif self.connection_string:
                if self.db_type == 'postgresql':
                    try:
                        import psycopg2  # type: ignore
                        self.db_connection = psycopg2.connect(self.connection_string)
                    except ImportError:
                        logger.warning("psycopg2 not installed. PostgreSQL connection unavailable.")
                elif self.db_type == 'mysql':
                    try:
                        import pymysql  # type: ignore
                        self.db_connection = pymysql.connect(self.connection_string)
                    except ImportError:
                        logger.warning("pymysql not installed. MySQL connection unavailable.")
            # Add more database types as needed
        except Exception as e:
            logger.warning(f"Could not initialize database connection: {e}")
            self.db_connection = None
    
    def _get_db_connection(self):
        """Get a database connection, creating a new one for thread safety"""
        try:
            if self.db_type == 'sqlite' and self.db_path:
                import sqlite3
                # Create a new connection for each execution to ensure thread safety
                # This avoids "SQLite objects created in a thread can only be used in that same thread" error
                return sqlite3.connect(self.db_path, check_same_thread=False)
            elif self.connection_string:
                if self.db_type == 'postgresql':
                    try:
                        import psycopg2  # type: ignore
                        return psycopg2.connect(self.connection_string)
                    except ImportError:
                        logger.warning("psycopg2 not installed. PostgreSQL connection unavailable.")
                        return None
                elif self.db_type == 'mysql':
                    try:
                        import pymysql  # type: ignore
                        return pymysql.connect(self.connection_string)
                    except ImportError:
                        logger.warning("pymysql not installed. MySQL connection unavailable.")
                        return None
            # Fallback to existing connection if available
            return self.db_connection
        except Exception as e:
            logger.warning(f"Could not create database connection: {e}")
            return self.db_connection if self.db_connection else None
    
    def reset(self, *args, **kwargs) -> None:
        """Reset tool state - clear cache and any internal state"""
        self.query_cache = {}
        logger.debug("RAGTextToSQL cache cleared")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query to extract only the current query intent.
        This helps remove context from previous queries that might have been mixed in.
        
        Handles cases like:
        - "top 5 users who watched The Lion King and also movie Twister" -> "top 5 users who watched movie Twister"
        - "users who watched movie X OR movie Y" -> extract only the last mentioned movie
        """
        if not query:
            return query
        
        import re
        
        # Remove common context mixing patterns
        # Pattern: "query about X" or "similar to previous query about X"
        query = re.sub(r'(similar to|like|same as|previous|previous query|old query).*?(?=\b(user|item|movie|film|phim|top|find|list|select)\b)', '', query, flags=re.IGNORECASE)
        
        # Detect if query contains multiple movie/item mentions that might be from different queries
        # Look for patterns like "movie X AND movie Y" or "movie X OR movie Y" or "movie X and also movie Y"
        movie_pattern = r'\b(movie|film|phim|title)\s+([^\s]+(?:\s+[^\s]+){0,3})'
        movies = re.findall(movie_pattern, query, re.IGNORECASE)
        
        # If multiple movies are mentioned, they might be from different queries
        # Common patterns: "AND", "OR", "and also", "additionally"
        if len(movies) > 1:
            # Check if there are separators indicating multiple queries
            separators = ['and also', 'additionally', 'also', 'or', 'and']
            query_lower = query.lower()
            
            # Find separators
            separator_positions = []
            for sep in separators:
                # Look for separator followed by movie/film keyword
                pattern = rf'\b{re.escape(sep)}\s+(movie|film|phim|title|the)'
                matches = list(re.finditer(pattern, query_lower, re.IGNORECASE))
                separator_positions.extend([m.start() for m in matches])
            
            # If we found separators, extract only the part after the last separator
            # This should be the most recent query
            if separator_positions:
                last_separator_pos = max(separator_positions)
                # Extract from separator, but include a bit before for context (e.g., "top 5 users")
                # Look backwards for query start keywords
                query_start_keywords = ['top', 'find', 'list', 'who', 'what', 'which', 'users', 'user']
                query_start_pos = 0
                
                # Find the nearest query start keyword before the separator
                for keyword in query_start_keywords:
                    pattern = rf'\b{re.escape(keyword)}\s+'
                    matches = list(re.finditer(pattern, query[:last_separator_pos], re.IGNORECASE))
                    if matches:
                        query_start_pos = max(query_start_pos, matches[-1].start())
                
                # Extract the query part after separator (or from query start if found)
                if query_start_pos > 0:
                    query = query[query_start_pos:]
                else:
                    # Remove everything before the last separator
                    query = query[last_separator_pos:]
                    # Remove the separator itself
                    query = re.sub(r'^(and also|additionally|also|or|and)\s+', '', query, flags=re.IGNORECASE)
        
        # Clean up: remove extra whitespace
        query = ' '.join(query.split())
        
        logger.debug(f"Normalized query: {query[:100]}...")
        return query
    
    def _extract_key_terms(self, query: str) -> str:
        """
        Extract key terms from query for better RAG retrieval.
        Focuses on the main entities and actions, ignoring context words.
        """
        if not query:
            return query
        
        # Key database-related terms
        db_terms = [
            'user', 'item', 'movie', 'film', 'phim', 'interaction', 'rating',
            'top', 'list', 'find', 'select', 'who', 'what', 'which',
            'watched', 'xem', 'đã xem', 'viewed', 'rated', 'đánh giá'
        ]
        
        # Extract terms that are likely the main query
        words = query.lower().split()
        key_terms = []
        
        for word in words:
            # Include if it's a database term or looks like an entity (capitalized, quoted, etc.)
            if any(term in word for term in db_terms) or word.isdigit():
                key_terms.append(word)
        
        # If we extracted key terms, use them; otherwise use original query
        if key_terms:
            extracted = ' '.join(key_terms)
            logger.debug(f"Extracted key terms: {extracted}")
            return extracted
        
        return query
    
    def _retrieve_schema_context(self, query: str) -> str:
        """Use RAG to retrieve relevant schema context"""
        if not self.vector_store:
            return "No schema documentation available."
        
        try:
            # Normalize query to remove mixed context from previous queries
            normalized_query = self._normalize_query(query)
            
            # Extract key terms for better RAG retrieval
            # This helps focus on the main query intent rather than all the context
            search_query = self._extract_key_terms(normalized_query)
            
            # If extraction produced a very short query, use normalized query instead
            if len(search_query.split()) < 3:
                search_query = normalized_query
            
            logger.debug(f"RAG search query: {search_query[:100]}...")
            
            # Search for relevant schema information using focused query
            docs = self.vector_store.similarity_search(search_query, k=self.top_k)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            if context:
                logger.debug(f"Retrieved {len(docs)} schema documents")
            
            return context
        except Exception as e:
            logger.warning(f"Error retrieving schema context: {e}")
            return f"Error retrieving schema context: {e}"
    
    def _text_to_sql(self, query: str, schema_context: str) -> str:
        """Convert natural language query to SQL using LLM"""
        # Normalize query to extract only current query intent
        normalized_query = self._normalize_query(query)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert specializing in movie recommendation systems. Convert natural language queries to SQL.
Given the database schema context, generate a valid SQL query for movie recommendations.

CRITICAL RULES:
1. Focus ONLY on the current query provided. Ignore any mentions of previous queries, similar queries, or "and also" clauses that might reference other queries.
2. If the query mentions multiple items (e.g., "Lion King AND Twister"), only include items explicitly mentioned in the CURRENT query, not from previous context.
3. Only use tables and columns mentioned in the schema context
4. Return ONLY ONE SQL query, no explanations
5. Use proper SQL syntax
6. For SQLite, use appropriate functions (e.g., DATE() for dates)
7. If the query is ambiguous, make reasonable assumptions based on schema
8. IMPORTANT: When matching movie/item titles, use LIKE with wildcards (e.g., WHERE title LIKE '%Twister%') because titles may include years or additional text (e.g., "Twister (1996)")
9. For "top N" queries, use ORDER BY with appropriate criteria and LIMIT N
10. When querying users who watched/viewed items, JOIN interactions with items table on item_id
11. Use DISTINCT when listing unique users or items
12. For user-item queries, the typical pattern is: SELECT DISTINCT i.user_id FROM interactions i JOIN items it ON i.item_id = it.item_id WHERE it.title LIKE '%MovieName%'
13. CRITICAL: Return ONLY a single SQL statement. Do not return multiple SQL statements.
14. IMPORTANT: Only include entities explicitly mentioned in the CURRENT query. If the query says "top 5 users who watched Twister", do NOT include other movies like "Lion King" unless they are explicitly mentioned in this query.
15. VERY IMPORTANT: When two tables share the same column name (e.g., item_id in both interactions and items), ALWAYS qualify the column with the table alias (e.g., i.item_id or it.item_id) in SELECT, WHERE, GROUP BY, ORDER BY to avoid ambiguous column errors.

MOVIE RECOMMENDATION QUERIES:
For recommendation queries (e.g., "recommend movies for user_id = 1", "suggest films based on user preferences", "movies similar users liked"):
- Find user's favorite genres: SELECT genre FROM items WHERE item_id IN (SELECT item_id FROM interactions WHERE user_id = X AND rating >= 4)
- Find movies in user's favorite genres that user hasn't watched: SELECT item_id, title, genre FROM items WHERE genre LIKE '%Genre%' AND item_id NOT IN (SELECT item_id FROM interactions WHERE user_id = X)
- Collaborative filtering: Find movies liked by similar users (users who rated same movies highly): Use subqueries to find similar users, then find their high-rated movies
- Personalization based on age/gender/occupation: JOIN users table with interactions to filter by user attributes
- Typical recommendation pattern:
  SELECT it.item_id, it.title, it.genre, AVG(i.rating) as avg_rating, COUNT(DISTINCT i.user_id) as num_users
  FROM items it
  JOIN interactions i ON it.item_id = i.item_id
  WHERE it.item_id NOT IN (SELECT item_id FROM interactions WHERE user_id = X)
  AND (it.genre IN (SELECT DISTINCT genre FROM items WHERE item_id IN (SELECT item_id FROM interactions WHERE user_id = X AND rating >= 4))
       OR i.user_id IN (SELECT DISTINCT user_id FROM interactions WHERE item_id IN (SELECT item_id FROM interactions WHERE user_id = X AND rating >= 4) AND user_id != X))
  GROUP BY it.item_id
  ORDER BY avg_rating DESC, num_users DESC
  LIMIT 10

For genre-based recommendations:
- Find top movies in a genre:
  SELECT
    it.item_id,
    it.title,
    it.genre,
    AVG(i.rating) AS avg_rating
  FROM interactions i
  JOIN items it ON i.item_id = it.item_id
  WHERE it.genre LIKE '%Genre%'
  GROUP BY it.item_id, it.title, it.genre
  ORDER BY avg_rating DESC
  LIMIT N

For demographic-based recommendations:
- Movies popular among similar age/gender: SELECT it.item_id, it.title, AVG(i.rating) as avg_rating FROM interactions i JOIN items it ON i.item_id = it.item_id JOIN users u ON i.user_id = u.user_id WHERE u.age BETWEEN X-5 AND X+5 AND u.gender = 'Y' AND it.item_id NOT IN (SELECT item_id FROM interactions WHERE user_id = Z) GROUP BY it.item_id ORDER BY avg_rating DESC LIMIT N

Schema Context:
{schema_context}

User Query (CURRENT QUERY ONLY - ignore previous context): {query}

SQL Query:"""),
        ])
        
        try:
            prompt = prompt_template.format_messages(
                schema_context=schema_context,
                query=normalized_query  # Use normalized query to avoid mixing context
            )
            response = self.llm.invoke(prompt)
            sql = response.content.strip()
            
            # Remove markdown code blocks if present
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()
            
            # Split by semicolon and take only the first SQL statement
            # This ensures we only execute one SQL statement at a time
            if ';' in sql:
                sql_parts = [part.strip() for part in sql.split(';') if part.strip()]
                if sql_parts:
                    sql = sql_parts[0]
            
            # Remove any leading/trailing whitespace or newlines
            sql = sql.strip()
            
            return sql
        except Exception as e:
            return f"Error generating SQL: {e}"
    
    def _execute_sql(self, sql: str, return_df: bool = False):
        """Execute SQL query and return results (DataFrame or formatted string)"""
        # Get a connection (creates new one for thread safety, especially for SQLite)
        connection = self._get_db_connection()
        if not connection:
            message = "Database connection not available."
            return (None, message) if return_df else message
        
        try:
            # Ensure we only execute a single SQL statement
            # Split by semicolon and take only the first statement
            if ';' in sql:
                sql_parts = [part.strip() for part in sql.split(';') if part.strip()]
                if sql_parts:
                    sql = sql_parts[0]
            
            # Remove any leading/trailing whitespace
            sql = sql.strip()
            
            # Validate that it's not empty
            if not sql:
                return "Error: Empty SQL statement."
            
            # Execute query with the connection
            if self.db_type == 'sqlite':
                df = pd.read_sql_query(sql, connection)
            else:
                df = pd.read_sql(sql, connection)
            
            # Close the connection if we created a new one (for SQLite)
            # This ensures thread safety by not reusing connections across threads
            if self.db_type == 'sqlite' and connection != self.db_connection:
                try:
                    connection.close()
                except Exception:
                    pass
            
            if df.empty:
                message = "Query executed successfully but returned no results."
                return (df, message) if return_df else message
            
            if return_df:
                return df, None
            
            # Format results
            result_str = f"Query returned {len(df)} rows:\n"
            result_str += df.to_string(index=False)
            
            # Limit output length
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "... (truncated)"
            
            return result_str
        except Exception as e:
            # Close connection on error if it's a new one
            if self.db_type == 'sqlite' and connection != self.db_connection:
                try:
                    connection.close()
                except Exception:
                    pass
            error_msg = f"Error executing SQL: {str(e)}"
            return (None, error_msg) if return_df else error_msg

    def _format_natural_response(self, user_query: str, sql: str, df: pd.DataFrame) -> str:
        """
        Convert SQL result DataFrame into a natural-language response tailored to the user query.
        """
        if df is None or df.empty:
            return "Không tìm thấy kết quả phù hợp với yêu cầu."
        
        preview_limit = self.config.get('response_row_limit', 5)
        preview_rows = df.head(preview_limit).fillna("").to_dict(orient="records")
        total_rows = len(df)
        
        table_preview = json.dumps(preview_rows, ensure_ascii=False, indent=2)
        has_more = total_rows > preview_limit
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful movie recommendation assistant. "
                       "Given a SQL result preview, answer the user in the same language as their query. "
                       "Be concise, natural, and focus on what the user asked. "
                       "If there are multiple rows, summarize the highlights rather than repeating raw data."),
            ("human", "User query:\n{user_query}\n\nSQL statement:\n{sql}\n\n"
                      "Result preview (first {preview_limit} of {total_rows} rows):\n{table_preview}\n\n"
                      "Write a natural-language answer describing the key findings. "
                      "If there are more rows than shown, mention that you are showing highlights.")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages(
                user_query=user_query,
                sql=sql,
                preview_limit=min(preview_limit, total_rows),
                total_rows=total_rows,
                table_preview=table_preview
            ))
            answer = response.content.strip()
            if not answer:
                raise ValueError("Empty response from LLM.")
            return answer
        except Exception as e:
            logger.warning(f"Failed to format natural response: {e}")
            # Fallback to a simple textual summary
            bullet_lines = []
            for row in preview_rows:
                title = row.get('title') or row.get('item_attributes') or row.get('item_id')
                meta = []
                if 'genre' in row and row['genre']:
                    meta.append(row['genre'])
                if 'avg_rating' in row and row['avg_rating'] != "":
                    meta.append(f"điểm TB {row['avg_rating']}")
                bullet = f"- {title}"
                if meta:
                    bullet += f" ({', '.join(meta)})"
                bullet_lines.append(bullet)
            extra_note = "\n... và còn nhiều kết quả khác." if has_more else ""
            fallback_response = "Một vài gợi ý nổi bật:\n" + "\n".join(bullet_lines) + extra_note
            return fallback_response
    
    def query(self, query: str, use_cache: bool = False) -> str:
        """
        Main method: Convert natural language to SQL and execute
        
        Args:
            query: Natural language query
            use_cache: Whether to use cached results. Defaults to False to ensure fresh results.
            
        Returns:
            SQL query result as string
        """
        # Clear cache before each new query to ensure fresh results
        # This prevents returning stale results from previous prompts
        if not use_cache:
            self.query_cache = {}
        
        # Check cache only if caching is enabled
        if use_cache and query in self.query_cache:
            logger.debug(f"Returning cached result for query: {query[:50]}...")
            return self.query_cache[query]
        
        logger.debug(f"Processing new query (cache disabled): {query[:50]}...")
        
        # Normalize query to extract only current query intent
        # This prevents mixing context from previous queries
        normalized_query = self._normalize_query(query)
        logger.debug(f"Normalized query: {normalized_query[:100]}...")
        
        # Step 1: Retrieve relevant schema context using RAG
        # Pass normalized query to avoid retrieving context for old queries
        schema_context = self._retrieve_schema_context(normalized_query)
        
        # Step 2: Convert to SQL
        # Pass both original and normalized query - LLM will use normalized but can reference original if needed
        sql = self._text_to_sql(normalized_query, schema_context)
        
        # Step 3: Execute SQL
        df, error_message = self._execute_sql(sql, return_df=True)
        if error_message:
            logger.warning("SQL execution failed: %s", error_message)
            return f"Không thể thực thi truy vấn hiện tại. Chi tiết: {error_message}"
        
        natural_response = self._format_natural_response(normalized_query, sql, df)

        # Extract item IDs if this looks like a recommendation query
        item_ids = self._extract_item_ids_from_df(df)
        if item_ids:
            natural_response += f'\n\nITEM_IDS: {",".join(map(str, item_ids))}'

        # Cache result only if caching is enabled
        if use_cache:
            self.query_cache[query] = natural_response

        return natural_response

    def _extract_item_ids_from_df(self, df: pd.DataFrame) -> List[int]:
        """
        Extract item IDs from DataFrame result
        """
        if df is None or df.empty:
            return []

        item_ids = []

        # Look for columns that might contain item IDs
        id_columns = ['item_id', 'id', 'movie_id', 'movieid', 'itemid']
        for col in id_columns:
            if col in df.columns:
                # Extract numeric IDs from this column
                for val in df[col].dropna():
                    try:
                        if isinstance(val, (int, float)) and val > 0:
                            item_ids.append(int(val))
                        elif isinstance(val, str):
                            # Try to extract number from string
                            import re
                            match = re.search(r'(\d+)', val)
                            if match:
                                item_ids.append(int(match.group(1)))
                    except:
                        continue

        # Remove duplicates and return
        return list(set(item_ids))

    def run_sql(self, sql: str) -> str:
        """
        Execute a raw SQL query directly (skips text-to-SQL step).
        """
        # Ensure we only execute a single SQL statement
        if ';' in sql:
            sql_parts = [part.strip() for part in sql.split(';') if part.strip()]
            if sql_parts:
                sql = sql_parts[0]
        
        sql = sql.strip()
        result = self._execute_sql(sql)
        return f"Executed SQL: {sql}\n\n{result}"
    
    def search(self, query: str) -> str:
        """
        Alias for query method to match RetrievalTool interface
        """
        return self.query(query)
    
    def lookup(self, title: str, term: str) -> str:
        """
        Not applicable for SQL tool, but required for RetrievalTool interface
        """
        return "Lookup method not supported for SQL tool. Use query() method instead."

