from setuptools import setup

setup(
    name='miso',
    version='2.0.5',
    packages=['miso', 'miso.data', 'miso.stats', 'miso.export', 'miso.layers', 'miso.models', 'miso.training'],
    install_requires=['image-classifiers', 'Keras', 'lxml', 'matplotlib', 'numpy', 'pandas', 'Pillow',
                      'scikit-image', 'scikit-learn', 'scipy', 'segmentation-models','dill'],
    url='',
    license='',
    author='Ross Marchant',
    author_email='ross.g.marchant@gmail.com',
    description=''
)
