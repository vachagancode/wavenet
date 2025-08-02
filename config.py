def get_config():
    hyperparameters = [
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
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 3,
            "epochs": 50
        },
        {
            "model_name": "m256k3v3",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 3,
            "epochs": 50
        },
        {
            "model_name": "m256k5v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "m256k5v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "m256k5v3",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
                {
            "model_name": "m256k7v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 7,
            "epochs": 50
        },
        {
            "model_name": "m256k7v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 7,
            "epochs": 50
        },
        {
            "model_name": "m256k7v3",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 7,
            "epochs": 50
        }
    ]

    return hyperparameters
