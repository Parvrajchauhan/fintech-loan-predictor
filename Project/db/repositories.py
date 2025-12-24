import pandas as pd
from sqlalchemy import text
from Project.db.sessions import get_engine


def save_dataframe(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "append"
):
    engine = get_engine()

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False
    )


def load_dataframe(
    table_name: str,
    columns: list | None = None,
    limit: int | None = None
) -> pd.DataFrame:
    engine = get_engine()

    cols = ", ".join(columns) if columns else "*"
    query = f"SELECT {cols} FROM {table_name}"

    if limit:
        query += f" LIMIT {limit}"

    return pd.read_sql(query, engine)



def load_single_row(
    table_name: str,
    index_id: int,
    columns: list | None = None
) -> pd.DataFrame:
    engine = get_engine()

    cols = ", ".join(columns) if columns else "*"
    query = f"""
        SELECT {cols}
        FROM {table_name}
        WHERE index_id = %(index_id)s
        LIMIT 1
    """

    return pd.read_sql(query, engine, params={"index_id": index_id})


