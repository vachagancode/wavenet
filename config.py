def get_config():
    hyperparameters = [
        {
            "model_name": "m256k1v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 1,
            "epochs": 50
        },
        {
            "model_name": "m256k1v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 1,
            "epochs": 50
        },
        {
            "model_name": "m256k1v3",
            "dconv_input": 64,
            "dconv_output": 64,
            "pconv_input": 64,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 1,
            "epochs": 50
        },
        {
            "model_name": "m256k2v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 2,
            "epochs": 50
        },
        {
            "model_name": "m256k2v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 2,
            "epochs": 50
        },
        {
            "model_name": "m256k2v3",
            "dconv_input": 64,
            "dconv_output": 64,
            "pconv_input": 64,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 2,
            "epochs": 50
        },
                {
            "model_name": "m256k3v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 3,
            "epochs": 50
        },
        {
            "model_name": "m256k3v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 3,
            "epochs": 50
        },
        {
            "model_name": "m256k3v3",
            "dconv_input": 64,
            "dconv_output": 64,
            "pconv_input": 64,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 3,
            "epochs": 50
        }
    ]

    return hyperparameters
