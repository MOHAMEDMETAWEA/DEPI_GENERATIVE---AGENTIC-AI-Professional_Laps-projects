import psycopg
from db import init_db
from config import PG_CONN_STR

print('init_db call')
init_db(PG_CONN_STR, 1536)
with psycopg.connect(PG_CONN_STR) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='rag_cv_chunks' ORDER BY ordinal_position;")
        cols = cur.fetchall()
print('cols:', cols)
