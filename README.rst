rnnr
====

*rnnr: neural network runner*

**rnnr** helps you to run your neural network models, for either training or evaluation.
It is heavily inspired by Ignite_ and torchnet_, but hopefully simpler and more applicable
to libraries other than PyTorch_.

Documentation
=============

https://rnnr.readthedocs.io

Contributing
============

Pull requests are welcome! To start contributing, make sure to install all the dependencies.

::

    pip install -r requirements.txt

Then install this library as editable package.

::

    pip install -e .

Next, setup the pre-commit hook.

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
