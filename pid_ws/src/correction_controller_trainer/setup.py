from setuptools import setup

package_name = "correction_controller_trainer"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="debug",
    maintainer_email="debug@example.com",
    description="State-selectable trajectory publisher and CSV logger for correction-controller training runs.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "scripted_trainer = correction_controller_trainer.scripted_trainer:main",
        ],
    },
)
