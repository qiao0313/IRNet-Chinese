import json

tables = json.load(open('data/db_schema.json', 'r', encoding='utf-8'))[0]
print(len(tables["column_names"]))
# 476