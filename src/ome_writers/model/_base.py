"""Base classes for OME-Writers models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenBaseModel(BaseModel):
    """Base model for all OME-Writers models with frozen configuration."""

    model_config = ConfigDict(frozen=True)
