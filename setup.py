from setuptools import setup

setup(name='color_naming',
      version='0.1',
      description='Implementation of ',
      url='',
      author='Filip Naise',
      author_email='filip@naiser.cz',
      license='MIT',
      packages=['color_naming'],
      install_requires=['numpy',],
      include_package_data=True,
      package_data={'color_naming': ['data/w2c.pkl']},
      zip_safe=False)
