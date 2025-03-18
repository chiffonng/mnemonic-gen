"""Initialize SQL engine and create tables for the database."""

from sqlmodel import SQLModel, create_engine

engine = create_engine("sqlite:///mnemonics.db")
SQLModel.metadata.create_all(engine)
