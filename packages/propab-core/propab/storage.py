from __future__ import annotations

from datetime import timedelta
from io import BytesIO
from minio import Minio
from minio.error import S3Error


def _parse_endpoint(endpoint: str) -> tuple[str, bool]:
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return "", False
    if endpoint.startswith("https://"):
        return endpoint.removeprefix("https://").split("/")[0], True
    if endpoint.startswith("http://"):
        return endpoint.removeprefix("http://").split("/")[0], False
    return endpoint, False


def get_minio_client() -> Minio | None:
    from propab.config import settings

    if not settings.minio_endpoint or not settings.minio_access_key or not settings.minio_secret_key:
        return None
    host, secure = _parse_endpoint(settings.minio_endpoint)
    if not host:
        return None
    use_https = secure or settings.minio_secure
    return Minio(
        host,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=use_https,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def put_bytes(*, object_name: str, data: bytes, content_type: str) -> str | None:
    client = get_minio_client()
    if client is None:
        return None
    from propab.config import settings

    bucket = settings.minio_bucket
    try:
        ensure_bucket(client, bucket)
        client.put_object(bucket, object_name, BytesIO(data), length=len(data), content_type=content_type)
        return client.presigned_get_object(bucket, object_name, expires=timedelta(days=7))
    except S3Error:
        return None


def put_text_file(*, object_name: str, text: str, content_type: str = "text/plain; charset=utf-8") -> str | None:
    return put_bytes(object_name=object_name, data=text.encode("utf-8"), content_type=content_type)
