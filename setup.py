from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='miso',
    version='4.0.0',
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
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=['lxml',
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'imagecodecs',
                      'scikit-image',
                      'scikit-learn',
                      'scipy',
                      'flask',
                      'tqdm',
                      'openpyxl',
                      'imblearn',
                      'tf2onnx',
                      'cleanlab',
                      'packaging',
                      'tensorflow_addons',
                      'flask_smorest',
                      'flask_cors',
                      'marshmallow_dataclass',
                      'celery[redis]',
                      'streamlit',
                      'dicttoxml',
                      'xmltodict'],
    url='https://github.com/microfossil/particle-classification',
    license='MIT',
    project_urls={
        'Source': 'https://github.com/microfossil/particle-classification',
        'Paper': 'https://jm.copernicus.org/articles/39/183/2020/',
    },
)
