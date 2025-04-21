.. TexTeller documentation master file, created by
   sphinx-quickstart on Sun Apr 20 13:05:53 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TexTeller Documentation
===========================================

Features
--------

- **Image to LaTeX Conversion**: Convert images containing LaTeX formulas to LaTeX code
- **LaTeX Detection**: Detect and locate LaTeX formulas in mixed text/formula images
- **Paragraph to Markdown**: Convert mixed text and formula images to Markdown format

Installation
-----------

You can install TexTeller using pip:

.. code-block:: bash

   pip install texteller

Quick Start
----------

Converting an image to LaTeX:

.. code-block:: python

   from texteller import load_model, load_tokenizer, img2latex

   # Load models
   model = load_model(use_onnx=False)
   tokenizer = load_tokenizer()

   # Convert image to LaTeX
   latex = img2latex(model, tokenizer, ["path/to/image.png"])[0]

Processing a mixed text/formula image:

.. code-block:: python

   from texteller import (
       load_model, load_tokenizer, load_latexdet_model,
       load_textdet_model, load_textrec_model, paragraph2md
   )

   # Load all required models
   latex_model = load_model()
   tokenizer = load_tokenizer()
   latex_detector = load_latexdet_model()
   text_detector = load_textdet_model()
   text_recognizer = load_textrec_model()

   # Convert to markdown
   markdown = paragraph2md(
       "path/to/mixed_image.png",
       latex_detector,
       text_detector,
       text_recognizer,
       latex_model,
       tokenizer
   )

API Documentation
----------------

For detailed API documentation, please see :doc:`./api`.

.. toctree::
   :maxdepth: 2
   :hidden:

   api
