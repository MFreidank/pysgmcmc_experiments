#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Sequence, ForeignKey
)


Base = declarative_base()


class Experiment(Base):
    """
    An experiment represents a series of `n_trial` trials.
    We track its unique `id`, the number of trials (`n_trials`) performed
    and the `date_time` of submission of the whole experiment.
    """

    __tablename__ = "experiments"

    # XXX How to generate a sequence properly here?
    experiment_id = Column(Integer, Sequence(), primary_key=True)

    n_trials = Column(Integer)

    date_time = Column(DateTime)


class Trial(Base):
    """ A trial represents a single run of a `sampler` with a given `stepsize`
        and sampler `parameters` on a given `benchmark`.
        We track the `time and date` of submission, and the resulting `samples` with
        their corresponding `costs` and most importantly:
        the value(s) of the target `metric` for this run.
    """
    __tablename__ = "trials"

    # XXX: How to generate a sequence properly here?
    trial_id = Column(Integer, Sequence(), primary_key=True)

    # XXX Ensure that this is how foreign keys are defined
    experiment_id = Column(ForeignKey(Experiment.experiment_id), nullable=False)

    stepsize = Column(Float, nullable=False)

    benchmark = Column(String, nullable=False)  # benchmark function used, e.g. banana

    sampler = Column(String, nullable=False)  # name of the sampler used, e.g. SGHMC

    parameters = Column(String)  # json encoded dictionary of all sampler?

    samples = Column(String)  # json encoded list of all samples from `benchmark`

    costs = Column(String)  # json encoded cost for each sample in samples

    metric_name = Column(String, nullable=False)  # name of the metric used.

    metric_value = Column(String, nullable=False)  # json encoded metric results

    date_time = Column(DateTime)


class TrialStatus(Base):
    """ A trial status represents status information where a trial for a particular experiment
        terminated unsucessfully. It represents a way to log errors in a
        visible and easily accessible way. This also allows us to selectively
        re-run only those experiments that failed.
    """

    __tablename__ = "trialerrors"

    trial_id = Column(Integer, Sequence())
