# meta-VQE does not need to be installed, but this will take care of the tequila installation
import setuptools

setuptools.setup(
    install_requires= ['tequila @ git+https://github.com/aspuru-guzik-group/tequila.git@master#egg=tequila' ]
)

