from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from backend.config import get_settings
from backend.db import Base

# Import models so their tables register on Base.metadata.
from backend.auth import models as _auth_models  # noqa: F401
from backend.projects import models as _project_models  # noqa: F401
from backend.clients import models as _client_models  # noqa: F401
from backend.runs import models as _run_models  # noqa: F401
from backend.trained_models import models as _trained_model_models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
