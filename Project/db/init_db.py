from sqlalchemy import text
from Project.db.sessions import get_engine

CREATE_REGRESSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS loan_regression_system_data (
);
"""

CREATE_CLASSIFICATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS loan_classification_system_data (
);
"""


def init_tables():
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(CREATE_REGRESSION_TABLE_SQL))
        conn.execute(text(CREATE_CLASSIFICATION_TABLE_SQL))


if __name__ == "__main__":
    init_tables()
