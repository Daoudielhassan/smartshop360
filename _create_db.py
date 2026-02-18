import psycopg2
c = psycopg2.connect(host='localhost', port=5432, dbname='postgres', user='postgres', password='postgres')
c.autocommit = True
cur = c.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname='smartshop_db'")
if not cur.fetchone():
    cur.execute("CREATE DATABASE smartshop_db")
    print("Base smartshop_db creee")
else:
    print("Base smartshop_db deja existante")
c.close()
