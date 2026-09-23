"""
SQLAlchemy ORM Data Models for Emotion Session Tracking and Event Persistence.
Supports PostgreSQL, MySQL, and SQLite.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class EmotionSession(Base):
    """Represents a continuous video or meeting session."""
    __tablename__ = "emotion_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_name = Column(String(128), nullable=False, default="Default Session")
    client_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    total_frames = Column(Integer, default=0)
    avg_valence = Column(Float, default=0.0)
    avg_arousal = Column(Float, default=0.0)
    dominant_emotion = Column(String(32), nullable=True)

    events = relationship("FaceDetectionEvent", back_populates="session", cascade="all, delete-orphan")


class FaceDetectionEvent(Base):
    """Represents a discrete face detection and emotion classification event."""
    __tablename__ = "face_detection_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("emotion_sessions.id"), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    face_idx = Column(Integer, default=0)
    bbox_x = Column(Integer, nullable=False)
    bbox_y = Column(Integer, nullable=False)
    bbox_w = Column(Integer, nullable=False)
    bbox_h = Column(Integer, nullable=False)
    dominant_emotion = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    valence = Column(Float, nullable=False)
    arousal = Column(Float, nullable=False)
    probabilities_json = Column(Text, nullable=True)

    session = relationship("EmotionSession", back_populates="events")
