from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_db_config():
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["database"]


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        db = get_db_config()
        _engine = create_engine(
            f"postgresql+psycopg2://{db['user']}:{db['password']}@"
            f"{db['host']}:{db['port']}/{db['name']}",
            pool_pre_ping=True,
        )
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal()