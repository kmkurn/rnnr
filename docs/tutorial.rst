Tutorial
========

.. currentmodule:: rnnr

To begin running a neural network training, first make sure that we have our iterable of
batches. We are free to use anything as a batch. For instance, if PyTorch_ is used, we may
represent the iterable of batches as a DataLoader_ object.

Once the batches are ready, we create a `Runner` object.

.. code-block:: python

   from rnnr import Runner
   trainer = Runner()

Next, we define ``batch_fn``, a function that is invoked on each batch of the run. For example,
if PyTorch_ is used, we can define our ``batch_fn`` like:

.. code-block:: python

   import torch.nn.functional as F

   def batch_fn(state):
       x, t = state['batch']
       y = model(x)
       loss = F.cross_entropy(y, t)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       state['output'] = loss.item()

To run the trainer, simply invoke the `~Runner.run` method.

.. code-block:: python

   trainer.run(batch_fn, batches, max_epoch=5)

Listening for events
--------------------

The above example is the minimum requirement to use a `Runner`. However, usually we want to
have some more control of what we do on the start of each epoch, end of each batch, and so on.
Heavily inspired by Ignite_, **rnnr** also uses an event system: during a run, the runner
emits events and we can provide callbacks for them.

.. code-block:: python

   from rnnr import Event

   @trainer.on(Event.STARTED)
   def print_training(state):
       print('Start training')

   @trainer.on(Event.EPOCH_STARTED)
   def print_epoch(state):
       print('Epoch', state['epoch'], 'started')

Now, when ``trainer`` is run, these callbacks will be invoked at the start of the run and the
start of each epoch respectively. The `~Runner.on` method can also be used *not* as a decorator
by passing a callback as its second argument.

.. code-block:: python

   def print_epoch_finished(state):
       print('Epoch', state['epoch'], 'of', state['max_epoch'], 'finished')

   trainer.on(Event.EPOCH_FINISHED, print_epoch_finished)

See `Runner` for more details. For more information on what events are available, see `Event`
instead. Please check :ref:`Callbacks` for some useful callbacks.

Callbacks that work together: an attachment
-------------------------------------------

An attachment is a collection of callbacks that work together to provide some functionality.
For example, to compute a mean over batch statistics, we need to append callbacks to many
events, and these callbacks work together to obtain the mean value. In **rnnr**, this concept
is realized by the `~attachments.Attachment` abstract base class. We can create our own
attachments, but some useful attachments are provided. See :ref:`Attachments` for more.


.. _PyTorch: https://pytorch.org
.. _DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
.. _Ignite: https://pytorch.org/ignite/index.html
