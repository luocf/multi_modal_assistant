import numpy as np
import pytest
import shutil
from app.core.identity.identity_manager import IdentityManager

def make_embedding(val):
    arr = np.ones(128, dtype=np.float64) * val
    arr = arr / np.linalg.norm(arr)
    return arr

@pytest.fixture
def idm(tmp_path):
    storage_dir = tmp_path / "test_identities"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    config = {
        'face_similarity_threshold': 0.0,
        'voice_similarity_threshold': 0.0,
        'feature_max_age': 3600,
        'max_features_per_user': 5,
        'storage_dir': str(storage_dir)
    }
    return IdentityManager(config)

def test_fuse_identities_face_voice(idm):
    face_emb = make_embedding(0.5)
    voice_emb = make_embedding(0.5)
    user_id = idm.create_profile(face_embedding=face_emb, voice_embedding=voice_emb)
    uid, conf = idm.fuse_identities(face_emb, voice_emb)
    assert uid == user_id
    assert conf['face'] >= 0.0 and conf['voice'] >= 0.0

def test_fuse_identities_multi(idm):
    face_emb = make_embedding(0.7)
    voice_emb = make_embedding(0.2)
    emotion = {'emotion': 'happy', 'confidence': 0.9}
    gesture = {'gesture': 'wave', 'confidence': 0.8}
    user_id = idm.create_profile(face_embedding=face_emb, voice_embedding=voice_emb)
    uid, conf = idm.fuse_identities(face_emb, voice_emb, emotion, gesture)
    assert uid == user_id
    assert conf['emotion'] > 0.5 and conf['gesture'] > 0.5

def test_update_profile(idm):
    face_emb = make_embedding(0.3)
    user_id = idm.create_profile(face_embedding=face_emb)
    new_voice = make_embedding(0.4)
    emotion = {'emotion': 'sad', 'confidence': 0.8}
    gesture = {'gesture': 'stop', 'confidence': 0.7}
    assert idm.update_profile(user_id, voice_embedding=new_voice, emotion=emotion, gesture=gesture)
    profile = idm.get_profile(user_id)
    assert profile.metadata['last_emotion']['emotion'] == 'sad'
    assert profile.metadata['last_gesture']['gesture'] == 'stop'

def test_save_load_delete(idm, tmp_path):
    face_emb = make_embedding(0.9)
    user_id = idm.create_profile(face_embedding=face_emb)
    idm.save_profiles(encrypt=True)
    idm.profiles.clear()
    idm.load_profiles(decrypt=True)
    assert user_id in idm.profiles
    idm.delete_user_profile(user_id)
    assert user_id not in idm.profiles 