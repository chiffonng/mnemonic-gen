"""Initialize SQL engine and create tables for the database."""

from pathlib import Path

from sqlmodel import SQLModel, create_engine

from src import const

data_dir = Path(const.PROCESSED_DATA_DIR).resolve()
engine = create_engine(f"sqlite:///{data_dir}/mnemonics.db")
SQLModel.metadata.create_all(engine)
