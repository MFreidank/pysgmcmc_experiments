from os.path import expanduser, join as path_join


PLOTS_DIRECTORY = path_join(
    expanduser("~"),
    "thesis_repos", "masterthesis_report", "sghmchd",
    "experiments", "results", "plots"
)

FONTSIZES = {
    "xlabel": 15,
    "ylabel": 15,
    "title": 15,
}
