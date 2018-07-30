from os.path import expanduser, join as path_join
from pylatex import Table
from pylatex.utils import NoEscape

TABLES_DIRECTORY = path_join(
    expanduser("~"),
    "thesis_repos", "masterthesis_report", "sghmchd",
    "experiments", "results", "tables"
)


def latex_table(inner_tabular: str, position="H",
                caption=None, escape_caption=False, label=None,
                output_filepath=None):
    table = Table(position=position)
    table.append(NoEscape(inner_tabular))

    if caption:
        caption_ = caption if escape_caption else NoEscape(caption)
        table.add_caption(caption_)

    if label:
        table.append(NoEscape("\label{{tab:{label}}}".format(label=label)))

    if output_filepath:
        table.generate_tex(output_filepath)

    return table.dumps()
