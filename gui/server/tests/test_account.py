import json

def test_login(client):
    response = client.post(
        '/login', 
        data=json.dumps(dict(username='zilliz', password='123456')),
        content_type='application/json'
    )
    
    assert response.status_code == 200