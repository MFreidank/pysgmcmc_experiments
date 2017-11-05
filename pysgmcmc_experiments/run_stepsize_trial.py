#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import argparse
import tensorflow as tf
from itertools import islice
import json

from database_schema import write_to_database, TrialStatus, StepsizeTrial
from pysgmcmc.sampling import Sampler
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule
from pysgmcmc.diagnostics.objective_function import (
    banana_log_likelihood,
    negative_log_likelihood,
)


# XXX Extract single chain from `sampler` on `benchmark` with `stepsize`
# XXX and write the status and results to our database
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sampler"
    )

    parser.add_argument(
        "benchmark"
    )

    parser.add_argument(
        "stepsize", action="store_float"
    )

    parser.add_argument(
        "experiment_id", action="store_int"
    )

    parser.add_argument(
        "database"
    )

    parser.add_argument(
        "--n-samples", dest="n_samples",
        action="store_int", default=10000
    )

    args = parser.parse_args()

    try:
        sampling_method = {
            "RelativisticSGHMC": Sampler.RelativisticSGHMC,
            "SGHMC": Sampler.SGHMC,
            "SGLD": Sampler.SGLD,
        }[args.sampler]

        benchmark = {
            "banana": negative_log_likelihood(banana_log_likelihood),
        }[args.benchmark]

        graph = tf.Graph()

        with tf.Session(graph=graph) as session:
            # XXX: use objective function in here
            sampler = Sampler.get_sampler(
                sampling_method=sampling_method,
                stepsize_schedule=ConstantStepsizeSchedule(args.stepsize),
                session=session
            )

            session.run(tf.global_variables_initializer())

            samples, costs = [], []

            for sample, cost in islice(sampler, args.n_samples):
                samples.append(sample)
                costs.append(cost)

            # XXX: Store varnames in DB?

    except Exception as e:
        status = TrialStatus(
            experiment_id=args.experiment_id,
            status="ERROR",
            error=str(e)
        )
        write_to_database(database_value=status, database_url=args.database)

    else:
        status = TrialStatus(
            experiment_id=args.experiment_id,
            status="SUCCESS",
            error=None
        )

        # XXX: Construct trial

        trial = None
        write_to_database(
            database_values=(status, trial,),
            database_url=args.database
        )


if __name__ == "__main__":
    main()
