#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import argparse
import imp
from os.path import dirname, isfile, splitext, join as path_join
from subprocess import check_call


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
        help="Python interpreter to use. Defaults to `python3`.",
        default="python3"
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

    parser.add_argument(
        "--repeat", help="Repeat n times. Defaults to `1`.",
        default=1, type=int,
    )

    parser.add_argument(
        "--long-running",
        help="Flag to state that a job will have runtime > 1h 23 min.",
        action="store_true",
        dest="long_running"
    )

    parser.add_argument(
        "--log-dir",
        help="Directory to populate with logfiles. Defaults to ./cluster_wrapper/logs/",
        action="store",
        dest="log_directory",
        default=path_join(dirname(__file__), "cluster_wrapper", "logs")
    )

    parser.add_argument(
        "--run-before",
        help="Path to a script that should be executed prior to running the jobs (e.g. to set up an environment)."
             "Defaults to cluster_wrapper/cluster_environment.sh",
        action="store",
        dest="run_before",
        default=path_join(dirname(__file__), "cluster_wrapper", "cluster_environment.sh")
    )

    args = parser.parse_args()
    assert isfile(args.scriptpath), args.scriptpath

    grid_helper = path_join(dirname(__file__), "cluster_wrapper/", "grid_helper.py")

    script_name, _ = splitext(args.scriptpath)

    if args.job_filename is None:
        args.job_filename = "{}.jobs".format(script_name)

    experiment_module = imp.load_source("experiment_module_imported", args.scriptpath)

    try:
        experiment = experiment_module.experiment
    except AttributeError:
        raise  # XXX: better error handling

    queues = (
        "meta_gpu-tf.q",
        "meta_gpunode-rz.q",
        "meta_gpu-rz.q",
        "aad_core.q",
        "aad_pe.q",
        "meta_core.q",
        "test_core.q"
    )

    assert args.queue in queues
    print(args.job_filename)

    with open(args.job_filename, "w") as f:
        for _ in range(args.repeat):
            if args.head is not None:
                f.writelines(experiment.to_jobs(interpreter=args.interpreter, scriptpath=args.scriptpath)[:int(args.head)])
            else:
                f.writelines(experiment.to_jobs(interpreter=args.interpreter, scriptpath=args.scriptpath))

    # Submit jobs file to cluster queue

    command = [
        "python2",
        grid_helper,
        "-q", args.queue
    ]

    if args.long_running:
        command += ["-lr"]

    command += [
        "-l", args.log_directory,
        "--startup", args.run_before,
    ]

    command += [args.job_filename]
    check_call(command)


if __name__ == "__main__":
    main()
