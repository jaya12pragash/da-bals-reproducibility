# Microsoft Azure LLM inference trace 2023

The two CSV files in this directory were obtained from the Microsoft Azure
Public Dataset repository. They contain anonymized production invocation
timestamps, context-token counts, and generated-token counts for code and
conversation services.

Source: https://github.com/Azure/AzurePublicDataset

Documentation:
https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md

License: Creative Commons Attribution 4.0 International. The repository's
`LICENSE` text is reproduced in this directory.

Required attribution:

Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Inigo Goiri,
Saeed Maleki, and Ricardo Bianchini, "Splitwise: Efficient generative LLM
inference using phase splitting," Proceedings of the 51st Annual International
Symposium on Computer Architecture, 2024.

DA-BALS uses these records only for a hybrid trace replay. Production
timestamps and token-count ranks are trace-derived. Time scaling, deadlines,
service calibration, shared background work, and cloud-edge topology remain
modeled and are not Azure deployment measurements.
