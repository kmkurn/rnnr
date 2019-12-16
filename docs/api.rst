API reference
=============

.. currentmodule:: rnnr

Event
-----

.. autoclass:: Event

Runner
------

.. autoclass:: Runner
   :members:

.. _Callbacks:

Callbacks
---------

.. currentmodule:: rnnr.callbacks

.. autofunction:: start_epoch_timer

.. autofunction:: stop_epoch_timer

.. autofunction:: maybe_stop_early

.. autofunction:: checkpoint

.. autofunction:: save

.. _Attachments:

Attachments
-----------

.. currentmodule:: rnnr.attachments

.. autoclass:: Attachment
   :members:
   :show-inheritance:

.. autoclass:: EpochTimer

.. autoclass:: ProgressBar

.. autoclass:: LambdaReducer

.. autoclass:: MeanReducer
