import psycopg 
from langchain_postgres import PostgresChatMessageHistory
from config import CONN_STR


connect_database = psycopg.connect(CONN_STR)


def create_tables(table_name="chat_history_online_g1"):
    PostgresChatMessageHistory.create_tables(connect_database, table_name)
    return f"Table {table_name} created successfully."
    


def get_history_from_postgres(user_id):
    table_name = "chat_history_online_g1"
    history =  PostgresChatMessageHistory(
    table_name,
    user_id,
    sync_connection=connect_database,
    )
    return history