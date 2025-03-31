from typing import Tuple, List, Dict, Any
import json
import datetime
import re


class MakeTableUseJson:
    def __init__(
            self, schema_json_path:str,
            init_sql_path: str = "/docker-entrypoint-initdb.d/init.sql",
            columns_key:str = "columns", data_key:str = "data", name_key: str = 'name',
            type_key: str = 'type', null_key: str = 'null', unique_key: str = 'unique',
            pk_key: str = 'primary_key'
        ):
        """
        json file의 table 스키마를 이용하여 table을 생성한다.
        >>> Dockerfile과 jsonfile을 이용하여 postgreSQL로 table 생성 시 사용하는 class

        >>> callable method로 실행 시, json file 내 table의 이름으로 사용
            jtm_ins = JSON_TABLE_MAKER(schema_json_path='schema.json')
            jtm_ins(table_name='user')

        >>> json file의 예)
            {
                "user":{
                    "columns":[
                        {"name":"id", "type":"TEXT", "null":false, "unique":false, "primary_key":true},
                        {"name":"passwd", "type":"TEXT", "null":false, "unique":false}
                    ],
                    "data":[
                        {"id":"root", "passwd":"admin"},
                        {"id":"user", "passwd":"normal"}
                    ]
                }
            }

        Args:
            schema_json_path (str): schema json 파일의 경로
            init_sql_path (str, optional): initial sql 파일의 경로 (생성될 파일). Defaults to "/docker-entrypoint-initdb.d/init.sql".
            columns_key (str, optional): schema json 파일 내, columns의 key. Defaults to "columns".
            data_key (str, optional): schema json 파일 내, data의 key. Defaults to "data".
            name_key (str, optional): schema json 파일 내, name의 key. Defaults to 'name'.
            type_key (str, optional): schema json 파일 내, type의 key. Defaults to 'type'.
            null_key (str, optional): schema json 파일 내, null의 key. Defaults to 'null'.
            unique_key (str, optional): _description_. schema json 파일 내, unique의 key. to 'unique'.
            pk_key (str, optional): _description_. schema json 파일 내, primary key의 key. to 'primary_key'.
        """
        self.schema_json_path = schema_json_path
        self.init_sql_path = init_sql_path
        self.columns_key = columns_key
        self.data_key = data_key
        self.name_key = name_key
        self.type_key = type_key
        self.null_key = null_key
        self.unique_key = unique_key
        self.pk_key = pk_key


    def __call__(self, table_name: str):
        """
        json 내 table_name에 대하여 table을 생성한다

        Args:
            table_name (str): table의 이름
        """
        # 1. json 파일로부터 대상 데이터를 가지고 온다
        columns, data = self.get_schema_from_json(table=table_name)
        # 2. create table query 생성
        create_table_query = self.make_create_table_query(table_name=table_name, columns=columns)
        # 3. insert into 생성
        insert_into = self.make_insert_into(table_name=table_name, data=data)
        # 4. self.init_sql_path 파일 생성
        self.make_initial_table(create_query=create_table_query, insert_query=insert_into)


    def get_schema_from_json(
            self,
            table: str,
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        table의 schema와 data가 들어있는 json 파일로부터 대상 데이터를 가지고 온다다

        Args:
            json_path (str): schema json 파일의 경로
            table (str): table의 이름

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: columns, data의 데이터(dict)가 포함된 list
        """
        with open(self.schema_json_path) as f:
            schema = json.load(f)

        columns = schema[table][self.columns_key]
        # data_key가 존재하지 않으면, 빈 list를 가지고 온다
        data = schema[table].get(self.data_key, [])
        return columns, data


    def make_create_table_query(
            self,
            table_name: str,
            columns: List[Dict[str, Any]],
        ) -> str:
        """
        json file 내, table의 스키마에 대하여 table create query를 생성한다

        Args:
            table_name (str): table의 이름
            columns (List[Dict[str, Any]]): json file 내, column의 정보들이 들어 있는 dictionary

        Returns:
            str: 생성된 CREATE TABLE SQL 쿼리 문자열
        """
        col_def_list = []   # column의 정의(definition)가 쌓이는 list
        pk_col_list = []    # primary key인 column이 쌓이는 list

        # 각 column의 정보를 기반으로 컬럼 정의(column definition)에 대한 문자열을 생성한다
        for column in columns:

            # column definition을 생성하여 col_def_list에 쌓는다
            col_def_list.append(self._make_column_definition(column))

            # primary key 여부에 따라 pk_col_list에 컬럼명을 쌓는다
            if column.get(self.pk_key): pk_col_list.append(column[self.name_key])

        # CREATE 쿼리를 생성한다
        pk_query = f", PRIMARY KEY ({', '.join(pk_col_list)})" if len(pk_col_list) > 0 else ""
        return f"CREATE TABLE {table_name} ({', '.join(col_def_list)}{pk_query});"
    

    def _make_column_definition(self, column: Dict[str, Any]) -> str:
        """
        컬럼 정의들을 생성한다

        Args:
            column (Dict[str, Any]): columns의 element

        Returns:
            str: columns의 element에 대한 컬럼 정의
        """
        # 기본 컬럼 정의
        col_def = f"{column[self.name_key]} {column[self.type_key]}"
        # NOT NULL 추가 여부
        if not column[self.null_key]: col_def = f"{col_def} NOT NULL"
        # UNIQUE 추가 여부
        if column[self.unique_key]: col_def = f"{col_def} UNIQUE"
        return col_def


    def make_insert_into(self, table_name: str, data: List[Dict[str, Any]]) -> List[str]:
        """
        table에 입력될 data에 대한 query를 생성한다

        Args:
            table_name (str): table의 이름
            data (List[Dict[str, Any]]): table에 insert into할 데이터

        Returns:
            List[str]: table에 insert into로 들어갈 쿼리
        """
        insert_query = []
        if len(data) > 0 :
            for record in data:
                keys = ", ".join(record.keys())
                # value의 type을 고려하여 format 수정
                values = ", ".join([self._format_value(value) for value in record.values()])
                # 쿼리 생성
                query = f"INSERT INTO {table_name} ({keys}) VALUES ({values});"
                insert_query.append(query)
        return insert_query
    

    def _format_value(self, value: Any) -> str:
        """
        value의 format을 postgreSQL의 query 형태에 맞게 수정

        Args:
            value (Any): 입력 데이터

        Returns:
            str: format이 수정된 value
        """
        if isinstance(value, str):
            if re.search(pattern="'", string=value) is not None:
                value = value.replace("'", "''")
            return f"'{value}'"
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif value is None:
            return 'NULL'
        elif isinstance(value, datetime.datetime):
            return f"'{value.isoformat()}'"
        else:
            return str(value)
        

    def make_initial_table(self, create_query: str, insert_query: List[str]):
        """
        Dockerfile로 docker image를 생성함과 동시에 table을 생성하는 경우 해당 메서드 사용
        self.init_sql_path에 sql 파일을 생성하고, 해당 파일을 Dockerfile에서 실행하여 table 생성성

        Args:
            create_query (str): CREATE TABLE에 대한 query
            insert_query (List[str]): INSERT INTO에 대한 query
        """
        # query가 저장된 sql 파일 생성
        with open(self.init_sql_path, "w") as f:
            f.write(create_query + "\n")
            # insert query 추가
            if len(insert_query) > 0:
                for ins in insert_query:
                    f.write(ins + "\n")
