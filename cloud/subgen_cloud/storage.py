import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import quote


class LocalObjectStorage:
    def __init__(self, root, public_base_url):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")

    def _path(self, key):
        candidate = (self.root / key).resolve()
        if self.root != candidate and self.root not in candidate.parents:
            raise ValueError("Invalid storage key")
        return candidate

    def initiate_upload(self, key, upload_id):
        parts = self.root / ".multipart" / upload_id
        parts.mkdir(parents=True, exist_ok=True)
        (parts / "metadata.json").write_text(json.dumps({"key": key}), encoding="utf-8")
        return upload_id

    def part_url(self, upload_id, part_number, key=None):
        return f"{self.public_base_url}/v1/uploads/{quote(upload_id)}/parts/{part_number}"

    def write_part(self, upload_id, part_number, body):
        target = self.root / ".multipart" / upload_id / f"{part_number:05d}.part"
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.md5(usedforsecurity=False)
        with target.open("wb") as stream:
            if isinstance(body, bytes):
                stream.write(body)
                digest.update(body)
            else:
                while True:
                    block = body.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
        return digest.hexdigest()

    def complete_upload(self, key, upload_id, parts):
        part_root = self.root / ".multipart" / upload_id
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            for part in sorted(parts, key=lambda value: value["part_number"]):
                source = part_root / f"{int(part['part_number']):05d}.part"
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        shutil.rmtree(part_root, ignore_errors=True)

    def abort_upload(self, key, upload_id):
        shutil.rmtree(self.root / ".multipart" / upload_id, ignore_errors=True)

    def download(self, key, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._path(key), destination)

    def upload_file(self, source, key, content_type=None):
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def signed_download_url(self, key, expires_seconds=900):
        return f"{self.public_base_url}/v1/local-objects/{quote(key, safe='')}"

    def open_object(self, key):
        return self._path(key)

    def exists(self, key):
        return self._path(key).exists()

    def object_size(self, key):
        return self._path(key).stat().st_size

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)


class S3ObjectStorage:
    def __init__(self, settings):
        import boto3
        from botocore.config import Config

        self.bucket = settings.storage_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.storage_endpoint,
            region_name=settings.storage_region,
            aws_access_key_id=settings.storage_access_key,
            aws_secret_access_key=settings.storage_secret_key,
            config=Config(signature_version="s3v4"),
        )

    def initiate_upload(self, key, upload_id=None):
        return self.client.create_multipart_upload(Bucket=self.bucket, Key=key)["UploadId"]

    def part_url(self, upload_id, part_number, key):
        return self.client.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": int(part_number),
            },
            ExpiresIn=3600,
        )

    def complete_upload(self, key, upload_id, parts):
        self.client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={
                "Parts": [
                    {"PartNumber": int(part["part_number"]), "ETag": part["etag"]}
                    for part in sorted(parts, key=lambda value: value["part_number"])
                ]
            },
        )

    def abort_upload(self, key, upload_id):
        self.client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)

    def download(self, key, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(destination))

    def upload_file(self, source, key, content_type=None):
        extra = {"ContentType": content_type} if content_type else None
        self.client.upload_file(str(source), self.bucket, key, ExtraArgs=extra)

    def signed_download_url(self, key, expires_seconds=900):
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )

    def exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return False
            raise

    def object_size(self, key):
        return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=key)


def create_storage(settings):
    if settings.storage_backend == "s3":
        return S3ObjectStorage(settings)
    return LocalObjectStorage(settings.local_storage_dir, settings.public_base_url)
