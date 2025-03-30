"""Initialize SQL engine and create tables for the database."""

from pathlib import Path

from sqlmodel import SQLModel, create_engine

from src import const

data_dir = Path(const.PROCESSED_DATA_DIR).resolve()
db_uri = f"sqlite:///{data_dir}/mnemonics.db"
engine = create_engine(db_uri, echo=True)


# Create tables based on SQLModel classes
def init_db():
    """Initialize the database by creating all tables."""
    # Import Mnemonic model to ensure it's registered with SQLModel
    from src.reason.mnemonic_models import Mnemonic

    SQLModel.metadata.create_all(engine)

    return db_uri


init_db()
