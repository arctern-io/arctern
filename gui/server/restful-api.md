- /scope
    - GET /scope (not required)
        - request: None
        - response: all scope name
    - POST /scope
        - request: None or specific scope_id
        - response: error if found, else scope_id

- /scope/<scope_id>
    - DELETE /scope/<scope_id>
        - request: scope_id
        - response: error if not found, else success

- /command
    - POST /command
        - request: scope_id & python command
        - response: 
            - error if scope_id not found
            - success if execute specific command in specific scope without exception
            - 400 if any exception occurs

- /session
    - POST /session
        - request: detail in [session.json](./session.json)
        - response:
            - error if scope_id not found
            - error if no specific session name
            - error if session name already in specific scope
            - 400 if any exception occurs
            - success if create session successfully

- /session/<scope_id>/<session_name>
    - DELETE /session/<scope_id>/<session_name>
        - request: scope_id & session_name
        - response:
            - error if scope_id not found
            - error if session not in scope
            - 400 if any exception occurs
            - success if remove session successfully

- /loadv2
- /loadfile (all supported file format)
- /createtable
- POST /table/info return information
sql & file two api
    - POST /loadv2
        - request: detail in [load_v2.json](./load_v2.json)
        - response:
            - error if scope_id not found
            - error if no specific session name
            - error if session not in scope
            - 400 if any exception occurs
            - success if load table successfully

- /login token(required)
- /dbs
- /db/tables
- /db/table/info

- /query
    - POST /query
        - request: detail in [query.json](./query.json)
        - response:
            - error if scope_id not found
            - error if no specific session name
            - error if session not in scope
            - 400 if any exception occurs
            - result of query

- /pointmap
    - POST /pointmap
        - request: detail in [pointmap.json](./pointmap.json)
