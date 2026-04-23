"""Alembic environment — uses a sync Postgres URL (see DATABASE_URL_SYNC in .env.example)."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_url() -> str:
    raw = os.getenv("DATABASE_URL_SYNC") or os.getenv("DATABASE_URL", "")
    if not raw:
        raise RuntimeError("Set DATABASE_URL_SYNC or DATABASE_URL for Alembic migrations.")
    return raw.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
