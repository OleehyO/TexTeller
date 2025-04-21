API Reference
=============

This section provides detailed API documentation for the TexTeller package. TexTeller is a tool for detecting and recognizing LaTeX formulas in images and converting mixed text and formula images to markdown.

.. contents:: Table of Contents
   :local:
   :depth: 2


Image to LaTeX Conversion
-------------------------

.. autofunction:: texteller.api.img2latex

Paragraph to Markdown Conversion
------------------------------

.. autofunction:: texteller.api.paragraph2md

LaTeX Detection
---------------

.. autofunction:: texteller.api.detection.latex_detect

Model Loading
-------------

.. autofunction:: texteller.api.load_model
.. autofunction:: texteller.api.load_tokenizer
.. autofunction:: texteller.api.load_latexdet_model
.. autofunction:: texteller.api.load_textdet_model
.. autofunction:: texteller.api.load_textrec_model


KaTeX Conversion
----------------

.. autofunction:: texteller.api.to_katex
