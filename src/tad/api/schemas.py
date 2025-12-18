from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Event(BaseModel):
    Login_Timestamp: Any = Field(alias="Login Timestamp")
    User_ID: str = Field(alias="User ID")
    IP_Address: str = Field(alias="IP Address")
    Country: str
    Browser_Name_and_Version: str = Field(alias="Browser Name and Version")
    Device_Type: str = Field(alias="Device Type")
    Login_Successful: int = Field(alias="Login Successful")


class ScoreRequest(BaseModel):
    events: list[dict[str, Any]]
    top_k: int = 50
    return_per_model: bool = False


class ScoreItem(BaseModel):
    session_id: str
    score: float
    meta: dict[str, Any] = {}


class ScoreResponse(BaseModel):
    top: list[ScoreItem]
