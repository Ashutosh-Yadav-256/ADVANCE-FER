"""
Tests for Database Models and Persistence.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, EmotionSession, FaceDetectionEvent


@pytest.fixture
def db_session():
    # In-memory test SQLite engine
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_session_and_events_cascade(db_session):
    """Creating a session and associating events."""
    sess = EmotionSession(session_name="Interview Mock", client_id="tester_1")
    db_session.add(sess)
    db_session.commit()

    event = FaceDetectionEvent(
        session_id=sess.id,
        face_idx=0,
        bbox_x=100,
        bbox_y=100,
        bbox_w=150,
        bbox_h=150,
        dominant_emotion="happy",
        confidence=0.92,
        valence=0.75,
        arousal=0.45
    )
    db_session.add(event)
    db_session.commit()

    retrieved = db_session.query(EmotionSession).filter_by(id=sess.id).first()
    assert retrieved is not None
    assert len(retrieved.events) == 1
    assert retrieved.events[0].dominant_emotion == "happy"
