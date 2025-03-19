"""Initialize SQL engine and create tables for the database."""

from pathlib import Path

from sqlmodel import SQLModel, create_engine

from src import const

data_dir = Path(const.PROCESSED_DATA_DIR).resolve()
engine = create_engine(f"sqlite:///{data_dir}/mnemonics.db")


# Create tables based on SQLModel classes
def init_db():
    """Initialize the database by creating all tables."""
    from src._data_prep.mnemonic_schemas import Mnemonic

    SQLModel.metadata.create_all(engine)


init_db()
