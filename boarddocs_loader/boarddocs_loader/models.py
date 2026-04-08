"""Pydantic models for internal data representation."""

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field


class ItemRef(BaseModel):
    item_id: str
    item_order: str
    item_name: str


class CategoryRecord(BaseModel):
    category_id: str
    category_order: str
    category_name: str
    items: list[ItemRef] = Field(default_factory=list)


class MeetingRecord(BaseModel):
    meeting_id: str
    date: dt.date | None = None
    name: str
    slug: str
    source_url: str = ""
    committee_id: str | None = None
    unid: str | None = None
    scraped_at: dt.datetime | None = None
    format: Literal["structured", "flat", "empty"] = "flat"
    categories: list[CategoryRecord] = Field(default_factory=list)
    files_found: int | None = None
    dir_path: str = ""


class AgendaItem(BaseModel):
    subject: str
    category: str = ""
    type: str = "Information"
    has_attachments: bool = False


class ItemJsonRecord(BaseModel):
    item_id: str
    item_order: str
    item_name: str
    item_slug: str = ""
    item_url: str = ""
    links: list[dict[str, Any]] = Field(default_factory=list)
    inner_html: str = ""


class DocumentRecord(BaseModel):
    tenant_id: str
    external_id: str
    document_type: str
    title: str | None = None
    content_raw: str | None = None
    content_text: str | None = None
    source_url: str | None = None
    file_path: str | None = None
    meeting_date: dt.date | None = None
    committee_name: str | None = None
    meeting_id: str | None = None
    agenda_item_id: str | None = None
    processing_status: str = "pending"
    metadata: dict[str, Any] = Field(default_factory=dict)
