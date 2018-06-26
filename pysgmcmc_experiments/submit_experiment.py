#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import argparse
import imp
from os.path import isfile, splitext


class NoWrappedExperimentFound(ValueError):
    """ Raised if a script does not have a local variable called `experiment` that represents a wrapped sacred experiment. """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scriptpath",
        help="Path to sacred experiment script to submit.\n"
             "Must define a local variable called experiment "
             "containing an instance of `pysgmcmc_experiments.experiment_wrapper.SacredExperiment`."
    )
    parser.add_argument(
        "--job-filename",
        help="Name of the temporary file used to specify actual jobs to submit to cluster nodes."
             "Defaults to `SCRIPTNAME.jobs`.",
        default=None
    )
    parser.add_argument(
        "--interpreter",
        help="Python interpreter to use. Defaults to `/usr/bin/python3`.",
        default="/usr/bin/python3"
    )
    parser.add_argument(
        "--queue",
        help="Cluster queue to use. Defaults to `meta_core.q`.",
        default="meta_core.q"
    )

    parser.add_argument(
        "--head", help="Only submit first `n` jobs.",
        default=None
    )

    args = parser.parse_args()
    assert isfile(args.scriptpath), args.scriptpath

    script_name, _ = splitext(args.scriptpath)

    if args.job_filename is None:
        args.job_filename = "{}.jobs".format(script_name)

    experiment_module = imp.load_source("experiment_module_imported", args.scriptpath)

    try:
        experiment = experiment_module.experiment
    except AttributeError:
        raise  # XXX: better error handling

    with open(args.job_filename, "w") as f:
        if args.head is not None:
            f.writelines(experiment.to_jobs(interpreter=args.interpreter)[:int(args.head)])
        else:
            f.writelines(experiment.to_jobs(interpreter=args.interpreter))

    # XXX: Submit jobs file to cluster queue


if __name__ == "__main__":
    main()
