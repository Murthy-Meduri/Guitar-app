"""
Google Drive integration and local persistent storage for Sargam Strings.

Provides:
- Instant retrieval of previously processed songs (0.01s lookup)
- Automatic upload of processed song notes, tabs, and metadata to Google Drive folder ('Sargam Strings Tabs')
- Seamless fallback to local storage ('songs/' directory) when Google Drive credentials are not yet configured
- Support for both Service Account (service_account.json) and OAuth (credentials.json)
"""

import os
import io
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SONGS_DIR = os.path.join(BASE_DIR, "songs")
os.makedirs(SONGS_DIR, exist_ok=True)
CATALOG_FILE = os.path.join(SONGS_DIR, "catalog.json")

GDRIVE_FOLDER_NAME = "Sargam Strings Tabs"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _load_local_catalog() -> Dict[str, Any]:
    if os.path.exists(CATALOG_FILE):
        try:
            with open(CATALOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_local_catalog(catalog: Dict[str, Any]):
    try:
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save local catalog: {e}")


class GoogleDriveService:
    def __init__(self):
        self.service = None
        self.folder_id = None
        self._init_service()

    def _init_service(self):
        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials

            creds = None
            service_account_path = os.path.join(BASE_DIR, "service_account.json")
            credentials_path = os.path.join(BASE_DIR, "credentials.json")
            token_path = os.path.join(BASE_DIR, "token.json")

            if os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON"):
                info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
                creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            elif os.path.exists(service_account_path):
                creds = service_account.Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
            elif os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
            elif os.path.exists(credentials_path):
                from google_auth_oauthlib.flow import InstalledAppFlow
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
                with open(token_path, "w") as token:
                    token.write(creds.to_json())

            if creds:
                self.service = build("drive", "v3", credentials=creds)
                self.folder_id = self._get_or_create_folder()
                logger.info(f"Google Drive service initialized. Folder ID: {self.folder_id}")
            else:
                logger.info("No Google Drive credentials found. Running in local persistence mode.")
        except Exception as e:
            logger.warning(f"Google Drive initialization skipped or failed: {e}. Falling back to local persistence.")
            self.service = None
            self.folder_id = None

    def _get_or_create_folder(self) -> Optional[str]:
        if not self.service:
            return None
        try:
            query = f"name='{GDRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
            files = results.get("files", [])
            if files:
                return files[0]["id"]

            file_metadata = {
                "name": GDRIVE_FOLDER_NAME,
                "mimeType": "application/vnd.google-apps.folder"
            }
            folder = self.service.files().create(body=file_metadata, fields="id").execute()
            return folder.get("id")
        except Exception as e:
            logger.error(f"Error creating Google Drive folder: {e}")
            return None

    def is_connected(self) -> bool:
        return self.service is not None and self.folder_id is not None

    def save_song(self, song_id: str, title: str, song_data: dict, query_or_video_id: Optional[str] = None) -> dict:
        """Saves song both locally and to Google Drive (if connected)."""
        payload = {
            "id": song_id,
            "title": title,
            "query": query_or_video_id or title,
            "data": song_data
        }

        # 1. Save to local storage
        local_song_path = os.path.join(SONGS_DIR, f"{song_id}.json")
        try:
            with open(local_song_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write local song file: {e}")

        # Update local catalog
        catalog = _load_local_catalog()
        catalog[song_id] = {
            "id": song_id,
            "title": title,
            "query": (query_or_video_id or title).strip().lower(),
            "notes_count": len(song_data.get("notes", [])) if isinstance(song_data, dict) else 0,
            "drive_synced": False
        }

        # 2. Upload to Google Drive if connected
        drive_file_id = None
        if self.is_connected():
            try:
                from googleapiclient.http import MediaInMemoryUpload
                json_bytes = json.dumps(payload).encode("utf-8")
                media = MediaInMemoryUpload(json_bytes, mimetype="application/json")

                # Check if file already exists in Drive folder
                file_name = f"{song_id}.json"
                q = f"name='{file_name}' and '{self.folder_id}' in parents and trashed=false"
                res = self.service.files().list(q=q, fields="files(id)").execute()
                existing = res.get("files", [])

                if existing:
                    file_id = existing[0]["id"]
                    self.service.files().update(fileId=file_id, media_body=media).execute()
                    drive_file_id = file_id
                else:
                    meta = {
                        "name": file_name,
                        "parents": [self.folder_id]
                    }
                    new_file = self.service.files().create(body=meta, media_body=media, fields="id").execute()
                    drive_file_id = new_file.get("id")

                catalog[song_id]["drive_synced"] = True
                catalog[song_id]["drive_file_id"] = drive_file_id
            except Exception as e:
                logger.error(f"Failed to upload to Google Drive: {e}")

        _save_local_catalog(catalog)
        return {"song_id": song_id, "drive_synced": catalog[song_id].get("drive_synced", False)}

    def get_song(self, song_id: str) -> Optional[dict]:
        """Loads a song by ID from local cache or Google Drive."""
        local_path = os.path.join(SONGS_DIR, f"{song_id}.json")
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        if self.is_connected():
            try:
                file_name = f"{song_id}.json"
                q = f"name='{file_name}' and '{self.folder_id}' in parents and trashed=false"
                res = self.service.files().list(q=q, fields="files(id)").execute()
                files = res.get("files", [])
                if files:
                    file_id = files[0]["id"]
                    content = self.service.files().get_media(fileId=file_id).execute()
                    song_obj = json.loads(content.decode("utf-8"))
                    # Cache locally
                    with open(local_path, "w", encoding="utf-8") as f:
                        json.dump(song_obj, f, indent=2)
                    return song_obj
            except Exception as e:
                logger.error(f"Error fetching song from Google Drive: {e}")

        return None

    def find_cached_song(self, query_or_id: str) -> Optional[dict]:
        """Check if this video ID or song query has already been processed and cached."""
        target = query_or_id.strip().lower()
        catalog = _load_local_catalog()

        # Match exact ID
        if query_or_id in catalog:
            return self.get_song(query_or_id)

        # Match query substring or title
        for sid, meta in catalog.items():
            if target in meta.get("query", "") or target in meta.get("title", "").lower():
                cached = self.get_song(sid)
                if cached:
                    return cached
        return None

    def list_songs(self) -> List[dict]:
        """Returns catalog of all processed songs."""
        catalog = _load_local_catalog()
        return list(catalog.values())


drive_service = GoogleDriveService()
