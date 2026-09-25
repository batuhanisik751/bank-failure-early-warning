"""Panel: the bank-quarter table that every label and feature step builds on.

``exits`` derives one row per bank with its failure date (or non-failure exit date and
reason); ``build`` joins those onto ``financials_raw`` with institution attributes and the
point-in-time availability date.
"""
