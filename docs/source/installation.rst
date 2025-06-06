Installation
============


This is the recommended way to install Scikit-NeuroMSI.

Installing  with pip
^^^^^^^^^^^^^^^^^^^^

Make sure that the Python interpreter can load Scikit-NeuroMSI code.
The most convenient way to do this is to use virtualenv, virtualenvwrapper, and pip.

After setting up and activating the virtualenv, run the following command:

.. code-block:: console

   $ pip install scikit-neuromsi
   ...

That should be enough.



Installing the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you’d like to be able to update your Scikit-NeuroMSI code occasionally with the
latest bug fixes and improvements, follow these instructions:

Make sure that you have Git installed and that you can run its commands from a shell.
(Enter *git help* at a shell prompt to test this.)

Check out Scikit-NeuroMSI main development branch like so:

.. code-block:: console

   $ git clone https://github.com/renatoparedes/scikit-neuromsi.git
   ...

This will create a directory *scikit-neuromsi* in your current directory.

Then you can proceed to install with the commands

.. code-block:: console

   $ cd scikit-neuromsi
   $ pip install -e .
   ...
