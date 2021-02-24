from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='miso2',
    version='2.4.1',
    description='Python scripts for training CNNs for particle classification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ross Marchant',
    author_email='ross.g.marchant@gmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
    ],
    keywords='microfossil, cnn',
    python_requires='>=3.6, <4',
    packages=['miso', 'miso.data', 'miso.deploy', 'miso.layers', 'miso.models', 'miso.stats', 'miso.training', 'miso.utils'],
    install_requires=['image-classifiers>=1.0.0',
                      'lxml',
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'Pillow',
                      'imagecodecs',
                      'scikit-image',
                      'scikit-learn',
                      'scipy',
                      'segmentation-models',
                      'dill',
                      'flask',
                      'tqdm',
                      'openpyxl',
                      'imblearn'],
    url='https://github.com/microfossil/particle-classification',
    license='MIT',
    project_urls={  # Optional
        'Source': 'https://github.com/microfossil/particle-classification',
        'Paper': 'https://jm.copernicus.org/articles/39/183/2020/',
    },
)
