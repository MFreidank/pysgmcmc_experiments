from os.path import abspath, dirname, join as path_join


from sacred import Experiment
from sacred.observers import FileStorageObserver


def to_experiment(experiment_name, configurations, function, observers=None):
    if observers is None:
        observers = (
            FileStorageObserver.create(
                path_join(dirname(__file__), "..", "results", experiment_name)
            ),
        )

    class SacredExperiment(object):
        def __init__(self,):
            self.name = experiment_name
            # XXX: Assert configurations is iterable iterable of dicts: field_name: VALUETOPASS each dict representing one setting of one parameter, each inner iterable one configuration
            self.configurations = configurations
            self.experiment = Experiment(self.name)
            for observer in observers:
                self.experiment.observers.append(
                    observer
                )

        def to_jobs(self, interpreter="/usr/bin/python3", scriptpath=abspath(__file__),
                    configurations=None):
            def format_configuration(configuration):
                return " ".join(
                    "{parameter}={value}".format(parameter=parameter, value=value)
                    for parameter, value in configuration.items()
                )
            configurations_ = configurations or self.configurations
            return tuple(
                "{interpreter} {script} with {configuration}\n".format(
                    interpreter=interpreter,
                    script=scriptpath,
                    configuration=format_configuration(configuration)
                )
                for configuration in configurations_
            )

    wrapper = SacredExperiment()
    wrapper.main = function
    # XXX: Assert that there is a senseful relationship between wrapper.configurations and arguments of wrapper.main
    wrapper.main = wrapper.experiment.automain(wrapper.main)
    return wrapper
