from subprocess import check_output
import re


def package_versions():
    package_list = check_output(("pip", "list")).decode().split("\n")[2:]

    packages = []

    package_pattern = re.compile(
        "([a-zA-Z0-9-]+) +([^ ]+).*"
    )

    for package in package_list:
        match = package_pattern.search(package)
        if match:
            package, version = match.group(1), match.group(2)
            packages.append((package.strip(), version.strip()))
    return packages
