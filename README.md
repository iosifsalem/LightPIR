# LightPIR
Implementation of LightPIR from the IFIP Networking 2021 paper "LightPIR: Privacy-Preserving Route Discovery for Payment Channel Networks" by Krzysztof Pietrzak, Iosif Salem, Stefan Schmid, and Michelle Yeo.

- main LightPIR implementation in lightning-HD-factorized.py
- Lightning Network snapshots in snapshots/
- paper arXiv version (and contact details): https://arxiv.org/pdf/2104.04293.pdf

## abstract

Payment channel networks are a promising approach to improve the scalability of cryptocurrencies: they allow to perform transactions in a peer-to-peer fashion, along multi-hop routes in the network, without requiring consensus on the blockchain. However, during the discovery of cost-efficient routes for the transaction, critical information may be revealed about the transacting entities.

This paper initiates the study of privacy-preserving route discovery mechanisms for payment channel networks. In particular, we present LightPIR, an approach which allows a source to efficiently discover a shortest path to its destination without revealing any information about the endpoints of the transaction. The two main observations which allow for an efficient solution in LightPIR are that: (1) surprisingly, hub labelling algorithms – which were developed to preprocess “street network like” graphs so one can later efficiently compute shortest paths – also work well for the graphs underlying payment channel networks, and that (2) hub labelling algorithms can be directly combined with private information retrieval.

LightPIR relies on a simple hub labeling heuristic on top of existing hub labeling algorithms which leverages the specific topological features of cryptocurrency networks to further minimize storage and bandwidth overheads. In a case study considering the Lightning network, we show that our approach is an order of magnitude more efficient compared to a privacy-preserving baseline based on using private information retrieval on a database that stores all pairs shortest paths.
