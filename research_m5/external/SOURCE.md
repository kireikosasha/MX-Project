# Where the models come from

`protocols/NN-name/family.py` are made by `mealy_dot.py` from the `.dot` files of the AALpy collection
(https://github.com/DES-Lab/AALpy, folder `DotModels`, commit b3ed2fb48cb77840fb4cfa6bf7d8aab50c70d18a, MIT licence,
copyright Edi Muskardin, TU Graz and Silicon Austria Labs). They are Mealy machines that others learned from real
implementations: nine Bluetooth Low Energy devices, five TLS servers, five MQTT brokers, four TCP stacks. The
stochastic variants of the collection and its toy examples are left out; nothing else is.

A family holds the table of the machine as data and nothing of the collection's code.
