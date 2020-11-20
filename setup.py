from setuptools import setup

setup(
    name='miso',
    version='2.2.2',
    packages=['miso', 'miso.data', 'miso.deploy', 'miso.layers', 'miso.models', 'miso.stats', 'miso.training'],
    install_requires=['image-classifiers>=1.0.0', 'lxml', 'matplotlib', 'numpy', 'pandas', 'Pillow',
                      'scikit-image', 'scikit-learn', 'scipy', 'segmentation-models', 'dill', 'flask', 'tqdm', 'xlrd'],
    url='',
    license='',
    author='Ross Marchant',
    author_email='ross.g.marchant@gmail.com',
    description=''
)
