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

import sqlite3
from pathlib import Path

import os
dirpath = os.path.split(os.path.realpath(__file__))[0]
DB = dirpath + '/../data/account.db'


class Account:
    """
    the class don't use singleton because cannot be accessed across threads
    """

    def __init__(self):
        self.__connection = sqlite3.connect(DB)

    def __del__(self):
        self.__connection.close()

    def __create__(self):
        print('create account db')
        cursor = self.__connection.cursor()
        cursor.execute('''
        CREATE TABLE Account
        (
            ACCOUNT TEXT PRIMARY KEY NOT NULL,
            PASSWORD TEXT
        );
        ''')
        cursor.close()
        self.__connection.commit()

    def __insert__(self, account, password):
        print("add account:{} password:{}".format(account, password))
        cursor = self.__connection.cursor()
        cursor.execute(
            'INSERT INTO Account (ACCOUNT, PASSWORD) VALUES ("{}", "{}");'
            .format(account, password))
        cursor.close()
        self.__connection.commit()

    def get_password(self, account):
        """
        get password by account name
        """
        cursor = self.__connection.cursor()
        rows = cursor.execute(
            'SELECT PASSWORD FROM Account WHERE ACCOUNT="{}";'.format(account))

        result = ''
        find = False
        for row in rows:
            result = row[0]
            find = True
            break
        cursor.close()
        return find, result


def init():
    """
    create account database
    """
    is_exist = Path(DB).exists()
    if not is_exist:
        for_init = Account()
        for_init.__create__()
        for_init.__insert__("zilliz", "123456")

    print("check/create account db")


init()
