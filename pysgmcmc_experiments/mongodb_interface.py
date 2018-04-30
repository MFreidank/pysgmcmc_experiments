import pymongo


def get_database(database_name):
    client = pymongo.MongoClient()
    return client.get_database(database_name)


def find_one_run(database_name):
    return get_database(database_name).runs.find_one()


def no_runs(database_name):
    return len(tuple(get_database("BNN_sinc").runs.find()))
