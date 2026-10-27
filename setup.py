from setuptools import find_packages, setup


setup(
    name='entregavel_7',
    version='0.0.1',
    packages=find_packages(where='entregavel_7'),
    package_dir={'': 'entregavel_7'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/entregavel_7']),
        ('share/entregavel_7', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Guilherme Terencio Machado e Jhonatan Ramos',
    maintainer_email='guilhermetm2@al.insper.edu.br',
    description='Segue linha e captura de creeper da APS 7.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'seguidor_de_linha = entregavel_7.seguidor_de_linha:main',
        'segue_linha = entregavel_7.segue_linha:main',
        'pega_creeper = entregavel_7.pega_creeper:main',
    ]},
)