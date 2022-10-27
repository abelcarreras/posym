from setuptools import setup, Extension
import numpy

include_dirs_numpy = [numpy.get_include()]

def get_version_number():
    main_ns = {}
    for line in open('posym/__init__.py', 'r').readlines():
        if not(line.find('__version__')):
            exec(line, main_ns)
            return main_ns['__version__']

def check_compiler():
    import subprocess
    output = subprocess.Popen(['gcc'], stderr=subprocess.PIPE).communicate()[1]
    if b'clang' in output:
        return 'clang'
    if b'gcc' in output:
        return 'gcc'


if check_compiler() == 'clang':
    integrals = Extension('posym.integrals',
                          extra_compile_args=['-std=c99'],
                          include_dirs=include_dirs_numpy,
                          sources=['c/integrals.c'])

else:
    print ('openmp is used')
    integrals = Extension('posym.integrals',
                          extra_compile_args=['-std=c99', '-fopenmp'],
                          extra_link_args=['-lgomp'],
                          include_dirs=include_dirs_numpy,
                          sources=['c/integrals.c'])

permutations = Extension('posym.permutations',
                         extra_compile_args=['-std=c99'],
                         include_dirs=include_dirs_numpy,
                         sources=['c/permutations.c'])


setup(name='posym',
      version=get_version_number(),
      description='posym module',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Abel Carreras',
      url='https://github.com/abelcarreras/posym',
      author_email='abelcarreras83@gmail.com',
      packages=['posym',
                'posym.operations'],
      install_requires=['numpy', 'scipy', 'pandas', 'PyYaml'],
      license='MIT License',
      ext_modules=[integrals, permutations])
