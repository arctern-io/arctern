"""
Copyright (C) 2019-2020 Zilliz. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from flask_httpauth import HTTPTokenAuth
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from itsdangerous import BadSignature, SignatureExpired

from arctern_server.app import account
from arctern_server.app.common import utils

# -H "Authorization: Token <jws-token>"
AUTH = HTTPTokenAuth(scheme='Token')


def create(username, expire):
    """
    create a token for account
    """
    token = Serializer(secret_key="secret_key", expires_in=expire) \
        .dumps({"user": username}) \
        .decode("utf-8")
    return token


@AUTH.verify_token
def verify(token):
    """
    check whether the token is valid
    """
    try:
        data = Serializer(secret_key='secret_key').loads(token)
    except (BadSignature, SignatureExpired):
        return False

    if not utils.check_json(data, 'user'):
        return False
    exist, _ = account.Account().get_password(data['user'])

    if not exist:
        return False

    print("token success")

    return True
