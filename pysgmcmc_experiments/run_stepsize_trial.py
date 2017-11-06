#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import argparse
from datetime import datetime
import tensorflow as tf
from itertools import islice
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists, create_database

from database_schema import TrialStatus, StepsizeTrial, Base
from pysgmcmc.sampling import Sampler
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule
from pysgmcmc.diagnostics.objective_functions import (
    banana_log_likelihood,
    to_negative_log_likelihood,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sampler"
    )

    parser.add_argument(
        "benchmark"
    )

    parser.add_argument(
        "stepsize", action="store", type=float
    )

    parser.add_argument(
        "experiment_id", action="store", type=float,
    )

    parser.add_argument(
        "database"
    )

    parser.add_argument(
        "--n-samples", dest="n_samples", type=int,
        action="store", default=10**4
    )

    args = parser.parse_args()

    dtype = tf.float64


    try:
        sampling_method = {
            "RelativisticSGHMC": Sampler.RelativisticSGHMC,
            "SGHMC": Sampler.SGHMC,
            "SGLD": Sampler.SGLD,
        }[args.sampler]


        benchmark = {
            "banana": (
                to_negative_log_likelihood(banana_log_likelihood), (0.0, 0.0),
            ),
        }[args.benchmark]

        graph = tf.Graph()

        cost_fun, initial_guess = benchmark

        sampler_args = {
            "dtype": dtype,
            "cost_fun": cost_fun,
            "sampling_method": sampling_method,
            "batch_generator": None,
            "stepsize_schedule": ConstantStepsizeSchedule(args.stepsize),
            "seed": 1,
        }

        with tf.Session(graph=graph) as session:
            sampler = Sampler.get_sampler(
                params=[tf.Variable(guess, dtype=dtype) for guess in initial_guess],
                **sampler_args, session=session
            )

            session.run(tf.global_variables_initializer())

            samples, costs = [], []

            varnames = [variable.name for variable in tf.trainable_variables()]

            for sample, cost in islice(sampler, args.n_samples):
                samples.append(sample)
                costs.append(cost)

        sampler_args.update({"initial_guess": json.dumps(initial_guess)})

    except Exception as e:
        # populate experiment with an `experiment_id`
        engine = create_engine(args.database)

        assert database_exists(engine.url)

        Base.metadata.bind = engine

        session = sessionmaker(bind=engine, autoflush=True)()

        status = TrialStatus(
            experiment_id=args.experiment_id,
            status="ERROR",
            error=str(e),
        )

        session.add(status)
        session.commit()

    else:
        # populate experiment with an `experiment_id`
        engine = create_engine(args.database)

        assert database_exists(engine.url)

        Base.metadata.bind = engine

        session = sessionmaker(bind=engine, autoflush=True)()

        status = TrialStatus(
            experiment_id=args.experiment_id,
            status="SUCCESS",
            error=None
        )

        # populate status with `trial_id`
        session.add(status)
        session.commit()

        trial = StepsizeTrial(
            trial_id=status.trial_id,
            experiment_id=args.experiment_id,
            stepsize=args.stepsize,
            benchmark=args.benchmark,
            sampler=args.sampler,
            parameters=json.dumps(sampler_args, default=str),
            parameter_names=json.dumps(varnames),
            samples=json.dumps(samples),
            costs=json.dumps(costs),
            date_time=datetime.now(),
        )

        session.add(trial)
        session.commit()



if __name__ == "__main__":
    main()
