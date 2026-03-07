from collections.abc import AsyncGenerator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.settings import settings


database_url = settings.database_url
connect_args: dict = {}

if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

parts = urlsplit(database_url)
query = dict(parse_qsl(parts.query, keep_blank_values=True))
sslmode = query.pop("sslmode", None)
if sslmode and sslmode.lower() in {"require", "verify-ca", "verify-full"}:
    connect_args["ssl"] = True

# libpq-style parameters that asyncpg does not accept
query.pop("channel_binding", None)

database_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

engine = create_async_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
AsyncSessionLocal = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
