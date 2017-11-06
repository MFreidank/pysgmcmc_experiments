#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import argparse
import numpy as np
from datetime import datetime
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists, create_database

from database_schema import Experiment, Base
from os.path import dirname, realpath, join as path_join
SCRIPT_PATH = dirname(realpath(__file__))
sys.path.insert(0, path_join(SCRIPT_PATH, "..", "..", ".."))

# XXX: This script should:
# Write a jobs file containing all trials!
# Write experiment data to our DB
# submit the jobs file to a cluster queue

# A seperate script should be called run_trial.py and should
# execute a single trial (that is a single sampler run with args on a benchmark)
# and write the results and the resulting status to the DB
# in the case of error, status will indicate an error state and contain a full
# error trace.


def jobs(stepsizes, script_path, interpreter, benchmark, sampler, experiment_id, database, n_repetitions=1):
    # XXX Script path is "run_stepsize_trial" (full) path
    print("SAMPLER:", sampler)
    trial = ("{interpreter} {script} {sampler} {benchmark} {{stepsize}} {experiment_id} {database}\n".format(
            sampler=sampler, benchmark=benchmark,
            interpreter=interpreter, script=script_path,
            database=database,
            experiment_id=experiment_id
    ).format)

    jobs = []

    for stepsize in stepsizes:
        trials = [trial(stepsize=stepsize)] * n_repetitions
        jobs.extend(trials)
    return jobs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampler", dest="sampler", action="store", default="RelativisticSGHMC"
    )

    parser.add_argument(
        "--benchmark", dest="benchmark", action="store", default="banana"
    )

    parser.add_argument(
        "--n-repetitions", dest="n_repetitions", action="store", default=1,
        type=int
    )
    parser.add_argument(
        "--stepsize-min", dest="stepsize_min", type=float,
        action="store", default=0.01
    )

    parser.add_argument(
        "--stepsize-increment", dest="stepsize_increment",
        action="store", default=0.05, type=float,
    )

    parser.add_argument(
        "--stepsize-max", dest="stepsize_max",
        action="store", default=12.01, type=float,
    )

    parser.add_argument(
        "--script", dest="script_path", action="store",
        default=path_join(SCRIPT_PATH, "run_stepsize_trial.py")
    )
    parser.add_argument(
        "--interpreter", dest="interpreter", action="store",
        default="python3"
    )

    parser.add_argument(
        "--database", dest="database_path", action="store",
        default=path_join(SCRIPT_PATH, "stepsize_experiments.db")
    )

    parser.add_argument(
        "--jobfile", dest="jobfile", action="store",
        default=path_join(SCRIPT_PATH, "jobs.txt")

    )

    args = parser.parse_args()

    assert args.n_repetitions >= 1

    stepsize_min = args.stepsize_min
    stepsize_increment = args.stepsize_increment
    stepsize_max = args.stepsize_max

    assert stepsize_increment > 0.
    assert stepsize_min > 0.
    assert stepsize_max >= stepsize_min

    stepsizes = np.arange(
        args.stepsize_min, args.stepsize_max, args.stepsize_increment
    )

    n_stepsizes = len(stepsizes)

    n_trials = args.n_repetitions * n_stepsizes

    experiment = Experiment(
        n_trials=n_trials,
        n_repetitions=args.n_repetitions,
        date_time=datetime.now(),
    )

    # populate experiment with an `experiment_id`
    engine = create_engine(args.database_path)

    assert database_exists(engine.url)

    Base.metadata.bind = engine

    session = sessionmaker(bind=engine, autoflush=True)()
    session.add(experiment)
    session.commit()

    #  Construct trials/jobs and submit them to the cluster {{{ #

    cluster_jobs = jobs(
        experiment_id=experiment.experiment_id,
        stepsizes=stepsizes,
        script_path=args.script_path,
        sampler=args.sampler,
        benchmark=args.benchmark,
        interpreter=args.interpreter,
        n_repetitions=args.n_repetitions,
        database=args.database_path
    )

    with open(args.jobfile, "w") as f:
        f.writelines(cluster_jobs)

    # XXX Submit job to cluster

    #  }}} Construct trials/jobs and submit them to the cluster #


if __name__ == "__main__":
    main()
