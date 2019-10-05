rnnr
====

*rnnr: neural network runner*

.. image:: https://img.shields.io/pypi/pyversions/rnnr.svg?style=flat
   :target: https://img.shields.io/pypi/pyversions/rnnr.svg?style=flat
   :alt: Python versions

.. image:: https://img.shields.io/pypi/v/rnnr.svg?style=flat
   :target: https://pypi.org/project/rnnr
   :alt: PyPI project

.. image:: https://img.shields.io/travis/kmkurn/rnnr.svg?style=flat
   :target: https://travis-ci.org/kmkurn/rnnr
   :alt: Build status

.. image:: https://img.shields.io/readthedocs/rnnr.svg?style=flat
   :target: https://rnnr.readthedocs.io
   :alt: Documentation status

.. image:: https://img.shields.io/coveralls/github/kmkurn/rnnr.svg?style=flat
   :target: https://coveralls.io/github/kmkurn/rnnr
   :alt: Code coverage

.. image:: https://img.shields.io/pypi/l/rnnr.svg?style=flat
   :target: http://www.apache.org/licenses/LICENSE-2.0
   :alt: License

.. image:: https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg
   :target: http://spacemacs.org
   :alt: Built with Spacemacs

**rnnr** helps you to run your neural network models, for either training or evaluation.
It is heavily inspired by Ignite_ and torchnet_, but hopefully simpler and more applicable
to libraries other than PyTorch_.

Documentation
=============

https://rnnr.readthedocs.io

Contributing
============

Pull requests are welcome! To start contributing, first install flit_.

::

    pip install flit

Next, install this library and its dependencies in development mode.

::

    flit install --symlink

Lastly, setup the pre-commit hook.

::

    ln -s ../../pre-commit.sh .git/hooks/pre-commit

Tests and the linter can be run with ``pytest`` and ``flake8`` respectively. The latter also
runs ``mypy`` for type checking.

License
=======

Apache License, Version 2.0


.. _PyTorch: https://pytorch.org
.. _Ignite: https://pytorch.org/ignite/index.html
.. _torchnet: https://github.com/pytorch/tnt/
.. _flit: https://pypi.org/project/flit/
