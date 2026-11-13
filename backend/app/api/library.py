"""Library collections, assignment, and inbox organize."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.corpus.library import (
    accept_organize,
    assign_document,
    create_collection,
    overview,
    preview_organize,
    undo_batch,
)
from app.db.session import get_db
from app.models import Document
from app.models.user import User

router = APIRouter(tags=["library"])


class CollectionIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)


class AssignIn(BaseModel):
    collection_id: uuid.UUID | None = None


class OrganizeItem(BaseModel):
    doc_id: uuid.UUID
    collection_id: uuid.UUID


class OrganizeIn(BaseModel):
    assignments: list[OrganizeItem] = Field(default_factory=list)


def _doc(db: Session, doc_id: uuid.UUID, owner) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None or (owner is not None and doc.owner_id != owner):
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@router.get("/library")
def library_overview(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict:
    return overview(db, current_owner(user))


@router.post("/collections")
def collections_create(
    body: CollectionIn,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict:
    try:
        row = create_collection(db, owner_id=current_owner(user), name=body.name, description=body.description)
        db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": str(row.id), "name": row.name}


@router.post("/documents/{doc_id}/collection")
def documents_assign(
    doc_id: uuid.UUID,
    body: AssignIn,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict:
    owner = current_owner(user)
    doc = _doc(db, doc_id, owner)
    if body.collection_id is not None:
        from app.models.collection import Collection

        library = db.get(Collection, body.collection_id)
        if library is None or (owner is not None and library.owner_id != owner):
            raise HTTPException(status_code=404, detail="collection not found")
    assign_document(db, doc, collection_id=body.collection_id, source="user")
    db.commit()
    return {"doc_id": str(doc.id), "collection_id": str(doc.collection_id) if doc.collection_id else None}


@router.get("/library/organize")
def library_organize_preview(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict:
    return {"groups": preview_organize(db, current_owner(user))}


@router.post("/library/organize")
def library_organize_accept(
    body: OrganizeIn,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict:
    batch = accept_organize(
        db,
        current_owner(user),
        [item.model_dump() for item in body.assignments],
    )
    db.commit()
    return {"batch_id": str(batch.id)}


@router.post("/library/batches/{batch_id}/undo")
def library_undo(
    batch_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict:
    try:
        restored = undo_batch(db, current_owner(user), batch_id)
        db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"restored": restored}
