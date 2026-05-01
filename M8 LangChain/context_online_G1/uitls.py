import uuid


def generate_user_id():
    user_id = str(uuid.uuid4())
    return user_id