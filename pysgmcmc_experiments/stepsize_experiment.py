#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import argparse
import numpy as np
from datetime import datetime
import sys
from database_schema import Experiment, write_to_database
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


def jobs(stepsizes, script_path, interpreter, sampler, n_repetitions=1):
    # XXX Script path is "run_stepsize_trial" (full) path
    trial = "{interpreter} {script} {{stepsize}}\n".format(
        interpreter=interpreter, script=script_path
    ).format

    jobs = []

    for stepsize in stepsizes:
        trials = [trial(stepsize)] * n_repetitions
        jobs.extend(trials)
    return jobs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-repetitions", dest="n_repetitions", action="store_int", default=1
    )
    parser.add_argument(
        "--stepsize-min", dest="stepsize_min",
        action="store_float", default=0.01
    )

    parser.add_argument(
        "--stepsize-increment", dest="stepsize_increment",
        action="store_float", default=0.05
    )

    parser.add_argument(
        "--stepsize-max", dest="stepsize_max",
        action="store_float", default=12.01
    )

    parser.add_argument(
        "--script", dest="script_path", action="store",
        default=path_join(SCRIPT_PATH, "run_stepsize_trial.py")
    )

    parser.add_argument(
        "--database", dest="database_path", action="store",
        default=path_join(SCRIPT_PATH, "stepsize_experiments.db")
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
        args.stepsize_min, args.stepsize_max, args.stepsize_step
    )

    n_stepsizes = len(stepsizes)

    n_trials = args.n_repetitions * n_stepsizes

    experiment = Experiment(
        n_trials=n_trials,
        n_repetitions=args.n_repetitions,
        date_time=datetime.now(),
    )

    #  Construct trials/jobs and submit them to the cluster {{{ #

    cluster_jobs = jobs(
        stepsizes=stepsizes,
        script_path=None,  # XXX put correct info here
        sampler=None,
        interpreter=None,
        n_repetitions=args.n_repetitions
    )

    with open(args.jobfile, "w") as f:
        f.writelines(cluster_jobs)

    # XXX Submit job to cluster

    #  }}} Construct trials/jobs and submit them to the cluster #

    # Write experiment information to database

    write_to_database(
        database_values=(experiment,), database_url=None  # XXX database url
    )


if __name__ == "__main__":
    main()
