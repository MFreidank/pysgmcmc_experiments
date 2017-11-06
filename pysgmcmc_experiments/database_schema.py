#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey
)
from sqlalchemy_utils import database_exists, create_database


Base = declarative_base()


class Experiment(Base):
    """
    An experiment represents a series of `n_trial` trials.
    We track its unique `id`, the number of trials (`n_trials`) performed
    and the `date_time` of submission of the whole experiment.
    """

    __tablename__ = "experiments"

    # XXX How to generate a sequence properly here?
    experiment_id = Column(Integer, primary_key=True)

    n_trials = Column(Integer, nullable=False)
    n_repetitions = Column(Integer, nullable=False)

    date_time = Column(DateTime, nullable=False)


class TrialStatus(Base):
    """ A trial status represents status information where a trial for a particular experiment
        terminated unsucessfully. It represents a way to log errors in a
        visible and easily accessible way. This also allows us to selectively
        re-run only those experiments that failed.
    """

    __tablename__ = "trialerrors"

    trial_id = Column(Integer, primary_key=True)
    experiment_id = Column(ForeignKey(Experiment.experiment_id), nullable=False)
    status = Column(String, nullable=False)
    error = Column(String, nullable=True)


class StepsizeTrial(Base):
    """ A trial represents a single run of a `sampler` with a given `stepsize`
        and sampler `parameters` on a given `benchmark`.
        We track the `time and date` of submission, and the resulting `samples` with
        their corresponding `costs` and most importantly:
        the value(s) of the target `metric` for this run.
    """
    __tablename__ = "stepsize_trials"

    dummy_id = Column(Integer, primary_key=True)

    trial_id = Column(ForeignKey(TrialStatus.trial_id), nullable=False)

    experiment_id = Column(ForeignKey(Experiment.experiment_id), nullable=False)

    stepsize = Column(Float, nullable=False)

    benchmark = Column(String, nullable=False)  # benchmark function used, e.g. banana

    sampler = Column(String, nullable=False)  # name of the sampler used, e.g. SGHMC

    parameters = Column(String, nullable=False)  # json encoded dictionary of sampler parameters

    parameter_names = Column(String, nullable=False)  # json encoded list of all (tensorflow) target parameter names

    samples = Column(String)  # json encoded list of all samples from `benchmark`

    costs = Column(String)  # json encoded cost for each sample in samples

    date_time = Column(DateTime)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("database_path", help="Relative path to database.")
    parser.add_argument("--db-type", dest="db_type", default="sqlite")

    args = parser.parse_args()

    engine = create_engine("{db_type}:///{path}".format(
        db_type=args.db_type, path=args.database_path)
    )

    if not database_exists(engine.url):
        create_database(engine.url)
        Base.metadata.create_all(engine)
    else:
        print("Database at path: '{}' already exists!".format(engine.url))
        sys.exit(1)


if __name__ == "__main__":
    main()
