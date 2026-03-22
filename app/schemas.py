from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class EducationItem(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    graduation_year: Optional[str] = None


class WorkExperienceItem(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[str] = None


class ProjectItem(BaseModel):
    name: Optional[str] = None
    duration: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class ResumeParseResult(BaseModel):
    filename: str
    content_type: Optional[str] = None
    contact: ContactInfo = Field(default_factory=ContactInfo)
    education: List[EducationItem] = Field(default_factory=list)
    work_experience: List[WorkExperienceItem] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    transformer_model: Optional[str] = None
    raw_text_characters: int = 0

