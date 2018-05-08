import pymongo


def get_database(database_name):
    client = pymongo.MongoClient()
    return client.get_database(database_name)


def find_one_run(database_name):
    return get_database(database_name).runs.find_one()


def no_runs(database_name):
    return get_database(database_name).runs.count()


def delete_database(database_name):
    client = pymongo.MongoClient()
    client.drop_database(database_name)


def successful_runs(database_name):
    return get_database(database_name).runs.find({"status": "COMPLETED"})


def runs_with_configuration(database_name, configuration, successful_only=True):
    if successful_only:
        runs = successful_runs(database_name)
    else:
        runs = get_database(database_name).runs

    for run in runs:
        assert all(key in run["config"].keys() for key in configuration.keys())
        if all(run["config"][key] == value for key, value in configuration.items()):
            yield run
